import json
import boto3
from typing import List

# 상수 정의 - 테스트를 위해 작은 값으로 설정
PARENT_CHUNK_SIZE = 100
CHILD_CHUNK_SIZE = 50
CHUNK_OVERLAP = 10

class MockBedrockRuntime:
    """Bedrock Runtime API 호출을 모킹하는 클래스"""
    def invoke_model(self, modelId, contentType, accept, body):
        request_body = json.loads(body)
        messages = request_body.get('messages', [])
        user_message = messages[0].get('content', '') if messages else ''
        
        # 청크 내용 추출
        chunk_start = user_message.find('<chunk>')
        chunk_end = user_message.find('</chunk>')
        chunk_content = user_message[chunk_start+7:chunk_end].strip() if chunk_start > -1 and chunk_end > -1 else ''
        
        # 간단한 컨텍스트 생성
        context = f"[컨텍스트: 이 청크는 '{chunk_content[:30]}...'로 시작하는 내용에 대한 설명입니다]"
        
        # 응답 형식 생성
        response_body = {
            'content': [{'text': context}]
        }
        
        class MockResponse:
            def __init__(self, body):
                self.body_content = json.dumps(body).encode('utf-8')
            
            def read(self):
                return self.body_content
        
        return {'body': MockResponse(response_body)}

class ContextualChunker:
    def __init__(self, use_mock=False):
        if use_mock:
            self.bedrock_runtime = MockBedrockRuntime()
        else:
            # Initialize the Bedrock Runtime client in us-east-1 region
            self.bedrock_runtime = boto3.client(
                service_name='bedrock-runtime',
                region_name='us-east-1'  # US region (North Virginia)
            )
    
    def get_chunk_context(self, whole_document, chunk_content):
        # 템플릿에 문서와 청크 내용 적용
        user_prompt = f"""
        <document>
        {whole_document}
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

        try:
            # Call the Bedrock Runtime to invoke Claude 3.5 Sonnet
            response = self.bedrock_runtime.invoke_model(
                modelId='us.anthropic.claude-3-5-haiku-20241022-v1:0',
                contentType='application/json',
                accept='application/json',
                body=request_body_json
            )

            # Parse the response
            response_body = json.loads(response['body'].read())
            assistant_response = response_body['content'][0]['text']
            print(f"Assistant response: {assistant_response}")
            
            return assistant_response
        except Exception as e:
            print(f"Error calling Bedrock API: {e}")
            # 오류 발생 시 기본 컨텍스트 반환
            return f"[컨텍스트: 이 청크는 문서의 일부입니다]"
    
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
        
        print(f"텍스트를 {len(parent_chunks)}개의 부모 청크로 나누었습니다.")
        
        # 2. 각 parent chunk를 CHILD_CHUNK_SIZE로 나누고 컨텍스트 추가
        for p_idx, parent_chunk in enumerate(parent_chunks):
            print(f"부모 청크 {p_idx+1} 처리 중...")
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
                print(f"  자식 청크 생성: {start_idx}~{end_idx} ({len(child_chunk)} 글자)")
                
                # 3. 각 child chunk에 컨텍스트 정보 추가
                print("  Bedrock API 호출하여 컨텍스트 정보 가져오는 중...")
                context_info = self.get_chunk_context(parent_chunk, child_chunk)
                
                # 컨텍스트 정보와 함께 청크 저장
                contextualized_chunk = f"{context_info}\n\n{child_chunk}"
                contextualized_chunks.append(contextualized_chunk)
                print(f"contextualized_chunk: {contextualized_chunk}")
                print("  청크 처리 완료")
        
        return contextualized_chunks

def test_contextual_chunker(use_mock=False):
    """ContextualChunker 클래스를 테스트하는 함수"""
    # 더미 텍스트 파일 읽기
    with open('dummy_text.txt', 'r', encoding='utf-8') as f:
        dummy_text = f.read()
    
    print(f"더미 텍스트 길이: {len(dummy_text)} 글자, {len(dummy_text.split())} 단어")
    
    # ContextualChunker 인스턴스 생성
    chunker = ContextualChunker(use_mock=use_mock)
    
    # 청킹 실행
    print(f"PARENT_CHUNK_SIZE: {PARENT_CHUNK_SIZE}, CHILD_CHUNK_SIZE: {CHILD_CHUNK_SIZE}, CHUNK_OVERLAP: {CHUNK_OVERLAP}")
    print("청킹 시작...")
    chunks = chunker.chunk(dummy_text)
    
    # 결과 출력
    print(f"\n총 {len(chunks)}개의 청크로 나뉘었습니다:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- 청크 {i+1} ---")
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)
    
    return chunks

if __name__ == "__main__":
    # 실제 Bedrock API 호출 (use_mock=False)
    # 또는 모킹 사용 (use_mock=True)
    test_contextual_chunker(use_mock=False)
