import json
from abc import abstractmethod, ABC
from typing import List
from urllib.parse import urlparse
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

PARENT_CHUNK_SIZE = 1024
CHILD_CHUNK_SIZE = 512
CHUNK_OVERLAP = 30

class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        raise NotImplementedError()
        
class SimpleChunker(Chunker):
    def chunk(self, text: str) -> List[str]:
        words = text.split()
        return [' '.join(words[i:i+100]) for i in range(0, len(words), 100)]
    
class ContextualChunker(Chunker):
    # 청크에 컨텍스트 추가하기, input으로 parent와 child chunk를 전달하고 contextualized chunk를 반환
    def get_chunk_context(self, parent_document, chunk_content):
        # Initialize the Bedrock Runtime client in us-east-1 region
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'  # US region (North Virginia)
        )

        # 템플릿에 문서와 청크 내용 적용
        user_prompt = f"""
        <document>
        {parent_document}
        </document>
        Here is the chunk we want to situate within the whole document.
        <chunk>
        {chunk_content}
        </chunk>
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Mention the title of the header that this chunk belongs to, followed by a short succinct context.
        Answer in Korean."""

        # 시스템 프롬프트 설정
        system_prompt = """
        You're an expert at providing a succinct context, targeted for specific text chunks.

        <instruction>
        - Offer 1-5 short sentences that explain what specific information this chunk provides within the document.
        - Focus on the unique content of this chunk, avoiding general statements about the overall document.
        - Clarify how this chunk's content relates to other parts of the document and its role in the document.
        - If there's essential information in the document that backs up this chunk's key points, mention the details.
        </instruction>
        """

        # Create the request payload with system prompt
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "temperature": 0.7
        }

        # Convert the request body to JSON
        request_body_json = json.dumps(request_body)

        # Call the Bedrock Runtime to invoke Claude 3.5 Sonnet
        response = bedrock_runtime.invoke_model(
                modelId='us.anthropic.claude-3-5-haiku-20241022-v1:0',
                contentType='application/json',
                accept='application/json',
                body=request_body_json
        )


        # Parse the response
        response_body = json.loads(response['body'].read())
        assistant_response = response_body['content'][0]['text']

        return assistant_response
        
    def chunk(self, text: str) -> List[str]:
        """
        텍스트를 청크로 나누는 메서드:
        1. 텍스트를 PARENT_CHUNK_SIZE로 먼저 나눔
        2. 각 parent chunk를 CHILD_CHUNK_SIZE로 나눔
        3. 각 child chunk에 컨텍스트 정보를 추가

        Args:
            text (str): 청크로 나눌 원본 텍스트

        Returns:
            List[str]: 컨텍스트가 추가된 청크 목록
        """
        # 결과를 저장할 리스트
        contextualized_chunks = []

        # 1. 텍스트를 PARENT_CHUNK_SIZE로 나누기
        parent_chunks = []
        words = text.split()

        for i in range(0, len(words), PARENT_CHUNK_SIZE):
            end_idx = min(i + PARENT_CHUNK_SIZE, len(words))
            parent_chunk = ' '.join(words[i:end_idx])
            parent_chunks.append(parent_chunk)

        # 2. 각 parent chunk를 CHILD_CHUNK_SIZE로 나누고 컨텍스트 추가
        for parent_chunk in parent_chunks:
            parent_words = parent_chunk.split()

            for i in range(0, len(parent_words), CHILD_CHUNK_SIZE - CHUNK_OVERLAP):
                if i > 0:
                    start_idx = i - CHUNK_OVERLAP
                else:
                    start_idx = 0

                end_idx = min(i + CHILD_CHUNK_SIZE, len(parent_words))

                # 마지막 청크가 너무 작으면 이전 청크와 병합
                if end_idx - start_idx < CHILD_CHUNK_SIZE // 2 and i > 0:
                    continue

                child_chunk = ' '.join(parent_words[start_idx:end_idx])
                # 컨텍스트 정보 가져오기
                context_info = self.get_chunk_context(parent_chunk, child_chunk)

                # 컨텍스트 정보와 함께 청크 저장
                contextualized_chunk = f"{context_info}\n\n{child_chunk}"
                contextualized_chunks.append(contextualized_chunk)

        return contextualized_chunks

def lambda_handler(event, context):
    logger.debug('input={}'.format(json.dumps(event)))
    s3 = boto3.client('s3')

    # Extract relevant information from the input event
    input_files = event.get('inputFiles')
    input_bucket =  event.get('bucketName')

    
    if not all([input_files, input_bucket]):
        raise ValueError("Missing required input parameters")
    
    output_files = []
    chunker = SimpleChunker()
    # chunker = ContextualChunker()

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
            chunked_content = process_content(file_content, chunker)
            
            output_key = f"Output/{input_key}"
            
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
    
    return result
    

def read_s3_file(s3_client, bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response['Body'].read().decode('utf-8'))

def write_to_s3(s3_client, bucket, key, content):
    s3_client.put_object(Bucket=bucket, Key=key, Body=json.dumps(content))    

def process_content(file_content: dict, chunker: Chunker) -> dict:
    chunked_content = {
        'fileContents': []
    }
    
    for content in file_content.get('fileContents', []):
        content_body = content.get('contentBody', '')
        content_type = content.get('contentType', '')
        content_metadata = content.get('contentMetadata', {})
        
        words = content['contentBody']
        chunks = chunker.chunk(words)
        
        for chunk in chunks:
            chunked_content['fileContents'].append({
                'contentType': content_type,
                'contentMetadata': content_metadata,
                'contentBody': chunk
            })
    
    return chunked_content