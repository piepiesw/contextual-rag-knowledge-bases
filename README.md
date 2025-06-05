## Knowledge Bases Custom Chunking Guide

- [**사용자 지정 변환 Lambda 함수를 사용하여 데이터 수집 방법 정의**](https://docs.aws.amazon.com/ko_kr/bedrock/latest/userguide/kb-custom-transformation.html)
- API 명세 확인

## Blog Reference - Contextual RAG

- [**Amazon Bedrock기반에서 Contextual Retrieval 활용한 검색 성능 향상 및 실용적 구성 방안**](https://aws.amazon.com/ko/blogs/tech/amazon-bedrock-contextual-retrieval/)

## Sample code

### Notebook version Contextual RAG

- [Notebook code](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/genai/aws-gen-ai-kr/20_applications/20_contextual_rag/notebook/1_file_processor.ipynb)

### CSV version Knowledge Bases Custom Chunking

```python
import json
from typing import List
import boto3
import logging
import io

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def read_s3_file(s3_client, bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response['Body'].read().decode('utf-8'))

def write_to_s3(s3_client, bucket, key, content):
    s3_client.put_object(Bucket=bucket, Key=key, Body=json.dumps(content, ensure_ascii=False))

def read_csv_from_text(csv_text):
    # Use StringIO to treat the CSV text as a file object
    with io.StringIO(csv_text) as file:
        lines = file.readlines()
        header = lines[0].strip()  # Remove any trailing newline characters
        content = [line.strip() for line in lines[1:]]  # Strip newline characters from each line
    return header, content

def split_csv_list(header, content, rows_per_file):
    total_rows = len(content)
    total_files = (total_rows + rows_per_file - 1) // rows_per_file
    files_content = []

    for i in range(total_files):
        start_index = i * rows_per_file
        end_index = start_index + rows_per_file
        slice_content = " \r\n ".join(content[start_index:end_index])
        files_content.append(header + " \r\n " + slice_content)
    return files_content

def process_content(file_content: dict, rows: int) -> dict:
    chunked_content = {
        'fileContents': []
    }

    for content in file_content.get('fileContents', []):
        content_body = content.get('contentBody', '')
        content_type = content.get('contentType', '')
        content_metadata = content.get('contentMetadata', {})

        header, files_content = read_csv_from_text(content_body)
        chunks = split_csv_list(header, files_content, rows)

        for chunk in chunks:
            chunked_content['fileContents'].append({
                'contentType': content_type,
                'contentMetadata': content_metadata,
                'contentBody': chunk
            })

    return chunked_content

def lambda_handler(event, context):
    logger.debug('input={}'.format(json.dumps(event)))
    s3 = boto3.client('s3')

    # Extract relevant information from the input event
    input_files = event.get('inputFiles')
    input_bucket = event.get('bucketName')

    if not all([input_files, input_bucket]):
        raise ValueError("Missing required input parameters")

    output_files = []

    for input_file in input_files:
        content_batches = input_file.get('contentBatches', [])
        file_metadata = input_file.get('fileMetadata', {})
        original_file_location = input_file.get('originalFileLocation', {})

        processed_batches = []

        for batch in content_batches:
            input_key = batch.get('key')

            if not input_key:
                raise ValueError("Missing uri in content batch")

            # Read file from S3
            file_content = read_s3_file(s3, input_bucket, input_key)

            # Process content (chunking)
            chunked_content = process_content(file_content, 5)

            output_key = f"output/{input_key}"

            # Write processed content back to S3
            write_to_s3(s3, input_bucket, output_key, chunked_content)

            # Add processed batch information
            processed_batches.append({
                'key': output_key
            })

        # Prepare output file information
        output_file = {
            'originalFileLocation': original_file_location,
            'fileMetadata': file_metadata,
            'contentBatches': processed_batches
        }
        output_files.append(output_file)

    result = {'outputFiles': output_files}

    print("#result")
    print(result)

    return result

```
