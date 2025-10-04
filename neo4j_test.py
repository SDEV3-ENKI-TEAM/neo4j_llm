import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("[+] 임베딩 모델 로딩 완료")
except Exception as e:
    print(f"[!] 임베딩 모델 로딩 실패: {e}")
    exit(1)

VECTOR_DIMENSION = 384
VECTOR_INDEX_NAME = "trace_pattern_embeddings"


class Neo4jTraceProcessor:    
    def __init__(self, uri: str, user: str, password: str):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print(f"[+] Neo4j 연결 성공: {uri}")
        except Exception as e:
            print(f"[!] Neo4j 연결 실패: {e}")
            print("    1. Neo4j 서버가 실행 중인지 확인")
            print("    2. .env 파일의 NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD 확인")
            exit(1)

    def close(self):
        if self.driver:
            self.driver.close()
            print("\n[+] Neo4j 연결 종료")

    def _extract_trace_features(self, trace_data: Dict[str, Any]):  # Dict[str, Any]
        features = {
            'processes': set(),
            'event_types': set(),
            'network_patterns': [],
            'file_operations': [],
            'process_chains': []
        }

        for span in trace_data.get("spans", []):
            tags_dict = {tag['key']: tag['value'] for tag in span.get('tags', [])}

            # 프로세스 이미지
            if 'Image' in tags_dict:
                process_name = Path(tags_dict['Image']).name
                features['processes'].add(process_name)

            # 이벤트 타입
            if 'EventName' in tags_dict:
                features['event_types'].add(tags_dict['EventName'])

            # 네트워크 패턴 (IP:Port)
            if 'DestinationIp' in tags_dict and 'DestinationPort' in tags_dict:
                net_pattern = f"{tags_dict['DestinationIp']}:{tags_dict['DestinationPort']}"
                features['network_patterns'].append(net_pattern)

            # 프로세스 체인 (부모-자식 관계)
            for ref in span.get('references', []):
                if ref.get('refType') == 'CHILD_OF':
                    features['process_chains'].append(
                        (ref['spanID'], span['spanID'])
                    )

        return features

    def _create_embedding_text(self, features: Dict[str, Any]): # str
        # 특징 -> 임베딩 텍스트 변환
        parts = []

        # 프로세스 패턴
        if features['processes']:
            parts.append(f"processes: {', '.join(sorted(features['processes']))}")

        # 이벤트 타입
        if features['event_types']:
            parts.append(f"events: {', '.join(sorted(features['event_types']))}")

        # 네트워크 패턴 (중복 제거)
        if features['network_patterns']:
            unique_nets = list(set(features['network_patterns']))
            parts.append(f"network: {', '.join(sorted(unique_nets))}")

        # 프로세스 체인 수
        if features['process_chains']:
            parts.append(f"chain_depth: {len(features['process_chains'])}")

        return " | ".join(parts)

    def setup_database(self):
        # 데이터 베이스 초기화 / 인덱스 생성
        with self.driver.session(database="neo4j") as session:
            # 기존 제약조건 확인
            print("[...] 데이터베이스 제약조건 설정 중...")
            session.run("""
                CREATE CONSTRAINT trace_id IF NOT EXISTS
                FOR (t:Trace) REQUIRE t.traceID IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT event_span_id IF NOT EXISTS
                FOR (e:Event) REQUIRE e.spanID IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT process_guid IF NOT EXISTS
                FOR (p:Process) REQUIRE p.guid IS UNIQUE
            """)

            # 벡터 인덱스 생성
            print(f"[...] 벡터 인덱스 '{VECTOR_INDEX_NAME}' 생성 중...")
            try:
                session.run("""
                    CREATE VECTOR INDEX $indexName IF NOT EXISTS
                    FOR (t:Trace) ON (t.embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: $dimension,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """, indexName=VECTOR_INDEX_NAME, dimension=VECTOR_DIMENSION)
                print("[+] 데이터베이스 설정 완료")
            except Exception as e:
                print(f"[!] 벡터 인덱스 생성 실패: {e}")
                print("    Neo4j 버전이 5.0 이상인지 확인하세요.")

    def clear_database(self):
        # 기존 데이터 삭제
        print("[...] 기존 데이터 삭제 중...")
        with self.driver.session(database="neo4j") as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("[+] 데이터베이스 초기화 완료")

    def process_trace_file(self, file_path: str): # bool
        # 단일 트레이스 파일 처리
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                trace_data = json.load(f)

            trace_id = trace_data.get("traceID")
            if not trace_id:
                print(f"  [!] {Path(file_path).name}: traceID 없음")
                return False

            # 특징 추출 및 임베딩 생성
            features = self._extract_trace_features(trace_data)
            embedding_text = self._create_embedding_text(features)
            embedding = embedding_model.encode(embedding_text).tolist()

            # 데이터베이스에 저장
            with self.driver.session(database="neo4j") as session:
                session.execute_write(
                    self._create_trace_graph,
                    trace_data, 
                    features, 
                    embedding_text, 
                    embedding
                )

            print(f"  [+] {Path(file_path).name}: {trace_id} 저장 완료")
            return True

        except json.JSONDecodeError as e:
            print(f"  [!] {Path(file_path).name}: JSON 파싱 오류 - {e}")
            return False
        except Exception as e:
            print(f"  [!] {Path(file_path).name}: 처리 오류 - {e}")
            return False

    @staticmethod
    def _create_trace_graph(tx, trace_data, features, embedding_text, embedding):
        # 트레이스 및 관련 노드 생성
        trace_id = trace_data["traceID"]

        # Trace 노드 생성
        tx.run("""
            MERGE (t:Trace {traceID: $trace_id})
            SET t.embedding = $embedding,
                t.pattern = $pattern,
                t.process_count = $proc_count,
                t.event_count = $event_count
        """, 
            trace_id=trace_id,
            embedding=embedding,
            pattern=embedding_text,
            proc_count=len(features['processes']),
            event_count=len(features['event_types'])
        )

        # Span/Event 노드 및 관계 생성
        for span in trace_data.get("spans", []):
            span_id = span["spanID"]
            tags_dict = {tag['key']: tag['value'] for tag in span.get('tags', [])}

            # Event 노드 생성
            tx.run("""
                MERGE (e:Event {spanID: $span_id})
                SET e.operationName = $op_name,
                    e.startTime = $start_time,
                    e.duration = $duration,
                    e.eventName = $event_name
                WITH e
                MATCH (t:Trace {traceID: $trace_id})
                MERGE (e)-[:PART_OF]->(t)
            """,
                span_id=span_id,
                op_name=span.get("operationName"),
                start_time=span.get("startTime"),
                duration=span.get("duration"),
                event_name=tags_dict.get("EventName", ""),
                trace_id=trace_id
            )

            # Process 노드 생성 및 관계
            process_guid = tags_dict.get("ProcessGuid")
            if process_guid:
                tx.run("""
                    MERGE (p:Process {guid: $guid})
                    SET p.image = $image,
                        p.processId = $pid
                    WITH p
                    MATCH (e:Event {spanID: $span_id})
                    MERGE (p)-[:EXECUTED]->(e)
                """,
                    guid=process_guid,
                    image=tags_dict.get("Image", ""),
                    pid=tags_dict.get("ProcessId"),
                    span_id=span_id
                )

            # 네트워크 연결 정보
            dest_ip = tags_dict.get("DestinationIp")
            dest_port = tags_dict.get("DestinationPort")
            if dest_ip and dest_port:
                tx.run("""
                    MERGE (n:NetworkEndpoint {ip: $ip, port: $port})
                    WITH n
                    MATCH (e:Event {spanID: $span_id})
                    MERGE (e)-[:CONNECTED_TO {protocol: $protocol}]->(n)
                """,
                    ip=dest_ip,
                    port=dest_port,
                    protocol=tags_dict.get("Protocol", ""),
                    span_id=span_id
                )

            # 부모-자식 관계
            for ref in span.get("references", []):
                if ref.get("refType") == "CHILD_OF":
                    tx.run("""
                        MATCH (parent:Event {spanID: $parent_id})
                        MATCH (child:Event {spanID: $child_id})
                        MERGE (parent)-[:PARENT_OF]->(child)
                    """,
                        parent_id=ref["spanID"],
                        child_id=span_id
                    )

    def find_similar_traces(self, target_trace_id: str, threshold: float = 0.8, top_k: int = 3):
        # 유사도 검색
        print(f"\n{'='*70}")
        print(f"유사도 검색: '{target_trace_id}'")
        print(f"임계값: {threshold}, 상위 {top_k}개 결과")
        print('='*70)

        with self.driver.session(database="neo4j") as session:
            result = session.run("""
                MATCH (target:Trace {traceID: $target_trace_id})
                CALL db.index.vector.queryNodes($indexName, $top_k_query, target.embedding)
                YIELD node, score
                WHERE score >= $threshold AND node.traceID <> $target_trace_id
                RETURN node.traceID AS traceID,
                       node.pattern AS pattern,
                       node.process_count AS processCount,
                       node.event_count AS eventCount,
                       score AS similarity
                ORDER BY score DESC
                LIMIT $top_k
            """,
                indexName=VECTOR_INDEX_NAME,
                target_trace_id=target_trace_id,
                top_k_query=top_k + 5, 
                threshold=threshold,
                top_k=top_k
            )

            similar_traces = list(result)

            if not similar_traces:
                print(f"\n⚠ 유사도 {threshold} 이상인 트레이스를 찾지 못했습니다.")
                return []

            print(f"\n✓ {len(similar_traces)}개의 유사 트레이스 발견:\n")
            for i, record in enumerate(similar_traces, 1):
                print(f"{i}. Trace ID: {record['traceID']}")
                print(f"   유사도: {record['similarity']:.4f}")
                print(f"   프로세스 수: {record['processCount']}, 이벤트 수: {record['eventCount']}")
                print(f"   패턴: {record['pattern']}")
                print()

            return similar_traces


def scan_json_files(root_dir: str): # List[str]
    json_files = []
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"[!] 디렉토리를 찾을 수 없습니다: {root_dir}")
        return json_files
    
    for file_path in root_path.rglob("*.json"):
        json_files.append(str(file_path))
    
    return sorted(json_files)


def main():
    print("="*70)
    print("Neo4j 트레이스 분석 시스템")
    print("="*70)
    
    # Neo4j 연결
    processor = Neo4jTraceProcessor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # 데이터베이스 설정
    processor.setup_database()
    
    # 기존 데이터 삭제
    # processor.clear_database()  # 필요시 주석 해제
    
    # 데이터 파일 스캔
    data_folder = "./data"
    print(f"\n[1] 데이터 파일 스캔: {data_folder}")
    json_files = scan_json_files(data_folder)
    
    if not json_files:
        print(f"[!] JSON 파일을 찾을 수 없습니다.")
        processor.close()
        return
    
    print(f"[+] {len(json_files)}개 파일 발견")
    
    # 모든 파일 처리
    print(f"\n[2] 트레이스 데이터 처리 시작")
    success_count = 0
    for file_path in json_files:
        if processor.process_trace_file(file_path):
            success_count += 1
    
    print(f"\n[+] 처리 완료: {success_count}/{len(json_files)} 성공")
    
    # 유사도 검색 테스트
    print(f"\n[3] 유사도 검색 테스트")
    
    # 실제 존재하는 traceID로 변경 필요
    test_trace_id = "51a239b290c1e6e4662092070fd3724b"
    
    # 먼저 해당 트레이스가 존재하는지 확인
    with processor.driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (t:Trace)
            RETURN t.traceID AS traceID
            LIMIT 1
        """)
        first_trace = result.single()
        if first_trace:
            test_trace_id = first_trace["traceID"]
            print(f"[+] 테스트 대상 Trace: {test_trace_id}")
        else:
            print("[!] 저장된 Trace가 없습니다.")
            processor.close()
            return
    
    # 유사도 검색 실행
    processor.find_similar_traces(
        target_trace_id=test_trace_id,
        threshold=0.8,
        top_k=3
    )
    
    # 연결 종료
    processor.close()


if __name__ == "__main__":
    main()
