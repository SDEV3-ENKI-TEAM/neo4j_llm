import os
import json
from dotenv import load_dotenv
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

from trace_analyzer_lib import Neo4jProcessor, TraceAnalyzer

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Kafka 설정
# 추가 필요

if __name__ == "__main__":
    neo4j_processor = Neo4jProcessor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    analyzer = TraceAnalyzer(neo4j_processor.driver, OPENAI_API_KEY)
    
    neo4j_processor.setup_database()
    
    print(f"\n--- Kafka('{KAFKA_TOPIC_NAME}') 토픽 구독 시작 ---")
    
    try:
        consumer = KafkaConsumer(...) # (이전과 동일하게 설정)
        for message in consumer:
            trace_data = message.value
            trace_id = trace_data.get("traceID")
            if not trace_id: continue

            print(f"\n--- 새로운 Trace 수신: {trace_id} ---")
            
            # 1. DB에 저장
            neo4j_processor.process_trace_data(trace_data)
            
            # 2. 유사 트레이스 검색 
            similar_traces = analyzer.find_similar_traces(trace_data)
            
            # 3. llm 활용해서 분석
            analysis_result = analyzer.analyze_with_cot(trace_data, similar_traces)
            
            print(f"\n[🤖 LLM 분석 리포트 - {trace_id}]")
            print(analysis_result)

    except Exception as e:
        print(f"[!] 오류 발생: {e}")
    finally:
        neo4j_processor.close()
