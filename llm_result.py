import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pathlib import Path

load_dotenv()

# 환경 변수
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 설정
VECTOR_INDEX_NAME = "trace_pattern_embeddings"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


class TraceGraphRetriever:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        print("[+] Neo4j 연결 성공")

    def close(self):
        if self.driver:
            self.driver.close()

    def _extract_features_for_embedding(self, trace_data: Dict[str, Any]):
        features = {"processes": set(), "events": set(), "networks": []}

        for span in trace_data.get("spans", []):
            tags = {t["key"]: t["value"] for t in span.get("tags", [])}

            if "Image" in tags:
                from pathlib import Path

                features["processes"].add(Path(tags["Image"]).name)
            if "EventName" in tags:
                features["events"].add(tags["EventName"])
            if "DestinationIp" in tags and "DestinationPort" in tags:
                features["networks"].append(
                    f"{tags['DestinationIp']}:{tags['DestinationPort']}"
                )

        parts = []
        if features["processes"]:
            parts.append(f"processes: {', '.join(sorted(features['processes']))}")
        if features["events"]:
            parts.append(f"events: {', '.join(sorted(features['events']))}")
        if features["networks"]:
            parts.append(f"network: {', '.join(sorted(set(features['networks'])))}")

        return " | ".join(parts)

    def find_similar_traces(
        self, target_trace_data: Dict[str, Any], threshold: float = 0.8, top_k: int = 3
    ): # List[Dict[str, Any]]
        # 벡터 유사도 기반으로 유사 트레이스 검색 + 그래프 관계 정보 포함
        # 임베딩 생성
        embedding_text = self._extract_features_for_embedding(target_trace_data)
        target_embedding = embedding_model.encode(embedding_text).tolist()

        print(f"\n[검색] 유사도 {threshold} 이상, 상위 {top_k}개 검색 중...")

        with self.driver.session(database="neo4j") as session:
            result = session.run(
                """
                // 벡터 유사도 검색
                CALL db.index.vector.queryNodes($indexName, $top_k_plus, $target_embedding)
                YIELD node, score
                WHERE score >= $threshold
                
                // 관계 정보 조회
                OPTIONAL MATCH (node)<-[:PART_OF]-(e:Event)
                OPTIONAL MATCH (p:Process)-[:EXECUTED]->(e)
                OPTIONAL MATCH (e)-[conn:CONNECTED_TO]->(net:NetworkEndpoint)
                OPTIONAL MATCH (e)-[:PARENT_OF]->(child:Event)
                
                WITH node, score,
                     collect(DISTINCT {
                         event: e.eventName,
                         operation: e.operationName,
                         process: p.image,
                         network: net.ip + ':' + toString(net.port),
                         protocol: conn.protocol
                     }) as details,
                     count(DISTINCT child) as childCount
                
                RETURN node.traceID AS traceID,
                       node.pattern AS pattern,
                       node.process_count AS processCount,
                       node.event_count AS eventCount,
                       score AS similarity,
                       details,
                       childCount
                ORDER BY score DESC
                LIMIT $top_k
            """,
                indexName=VECTOR_INDEX_NAME,
                target_embedding=target_embedding,
                top_k_plus=top_k + 5,
                threshold=threshold,
                top_k=top_k,
            )

            traces = []
            for record in result:
                traces.append(
                    {
                        "traceID": record["traceID"],
                        "pattern": record["pattern"],
                        "similarity": float(record["similarity"]),
                        "processCount": record["processCount"],
                        "eventCount": record["eventCount"],
                        "details": record["details"],
                        "childCount": record["childCount"],
                    }
                )

            if traces:
                print(f"[+] {len(traces)}개 유사 트레이스 발견")
                for t in traces:
                    print(f"    - {t['traceID']}: 유사도 {t['similarity']:.4f}")
            else:
                print(f"[!] 유사도 {threshold} 이상인 트레이스 없음")

            return traces


class GPTTraceAnalyzer:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        print(f"[+] GPT API 연결 ({model})")

    def _format_target_trace(self, trace_data: Dict[str, Any]):
        """분석 대상 트레이스 포맷"""
        lines = [f"### 분석 대상 Trace: {trace_data.get('traceID', 'Unknown')}"]
        lines.append(f"총 Span 수: {len(trace_data.get('spans', []))}")
        lines.append("\n#### 주요 이벤트:")

        for i, span in enumerate(trace_data.get("spans", [])[:8], 1):
            tags = {t["key"]: t["value"] for t in span.get("tags", [])}
            lines.append(f"\n**Event {i}**")
            lines.append(f"- Operation: {span.get('operationName', 'N/A')}")
            lines.append(f"- Event Type: {tags.get('EventName', 'N/A')}")
            lines.append(f"- Process: {tags.get('Image', 'N/A')}")
            lines.append(f"- User: {tags.get('User', 'N/A')}")

            dest_ip = tags.get("DestinationIp")
            dest_port = tags.get("DestinationPort")
            if dest_ip and dest_port:
                lines.append(
                    f"- Network: {dest_ip}:{dest_port} ({tags.get('Protocol', 'N/A')})"
                )

        return "\n".join(lines)

    def _format_similar_traces(self, similar_traces: List[Dict[str, Any]]):
        # 유사 트레이스 포맷
        if not similar_traces:
            return "유사한 과거 사례 없음"

        lines = ["### 유사한 과거 트레이스 (참고용)"]

        for i, trace in enumerate(similar_traces, 1):
            lines.append(f"\n#### 유사 사례 {i}")
            lines.append(f"- Trace ID: {trace['traceID']}")
            lines.append(f"- 유사도 점수: {trace['similarity']:.4f}")
            lines.append(f"- 패턴: {trace['pattern']}")
            lines.append(
                f"- 프로세스 수: {trace['processCount']}, 이벤트 수: {trace['eventCount']}"
            )
            lines.append(f"- 프로세스 체인 깊이: {trace['childCount']}")

            # 주요 이벤트
            if trace["details"]:
                lines.append("- 주요 활동:")
                for detail in trace["details"][:5]:
                    if detail["event"]:
                        proc = (
                            detail["process"].split("\\")[-1]
                            if detail["process"]
                            else "Unknown"
                        )
                        lines.append(f"  * {detail['event']} by {proc}")

        return "\n".join(lines)

    def analyze_with_cot(
        self, target_trace: Dict[str, Any], similar_traces: List[Dict[str, Any]]
    ):

        print("\n[분석] GPT Chain-of-Thought 분석 시작...")

        # 프롬프트 구성
        system_prompt = """당신은 시스템 보안 및 행위 분석 전문가입니다.

트레이스 데이터를 분석할 때 다음 단계를 따르세요:
1. **패턴 식별**: 관찰된 프로세스, 이벤트, 네트워크 활동 패턴 파악
2. **유사성 비교**: 과거 유사 사례와 비교하여 공통점과 차이점 분석
3. **위험도 평가**: 보안 관점에서 위험 요소 식별 (정상/의심/위험)
4. **핵심 요약**: 전체 활동을 자연어로 명확하게 요약
5. **대응 방안**: 구체적이고 실행 가능한 조치 사항 제시

각 단계의 사고 과정을 명확히 보여주고, 결론은 간결하게 정리하세요."""

        user_prompt = f"""{self._format_target_trace(target_trace)}

{self._format_similar_traces(similar_traces)}

---

위 트레이스를 5단계 분석 프로세스에 따라 분석하고, 다음을 제공하세요:

1. **단계별 사고 과정** (각 단계별로 상세히)
2. **핵심 요약** (2-3문장으로 자연어 요약)
3. **보안 위험도** (낮음/보통/높음 + 이유)
4. **대응 방안** (우선순위별로 3-5개 구체적 조치)

명확하고 실무에 적용 가능한 분석을 제공하세요."""

        # GPT API 호출
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # 일관성을 위해 낮은 temperature
            max_tokens=2000,
        )

        analysis = response.choices[0].message.content
        print("[+] 분석 완료")

        return analysis


def load_trace_file(file_path: str):  # Dict[str, Any]
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def main():

    print("=" * 70)
    print("Graph RAG 트레이스 분석 시스템")
    print("=" * 70)

    if not OPENAI_API_KEY:
        print("\n[!] 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("    .env 파일에 다음을 추가하세요:")
        print("    OPENAI_API_KEY=sk-your-api-key")
        return

    # 1. Neo4j 검색기 초기화
    retriever = TraceGraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # 2. GPT 분석기 초기화
    analyzer = GPTTraceAnalyzer(OPENAI_API_KEY, model="gpt-4o")

    # 3. 분석할 트레이스 로드
    # 프로젝트(현재 파일 위치) 기준의 result 폴더를 보장하고, 예시 파일이 없으면 생성
    here = Path(__file__).resolve().parent
    result_dir = here / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    test_trace_path = result_dir / "trace-example.json"

    if test_trace_path.exists():
        print(f"\n[+] 트레이스 로드: {test_trace_path}")
        target_trace = load_trace_file(str(test_trace_path))
    else:
        # 파일이 없으면 예시 데이터를 생성해서 저장
        print(f"\n[!] 파일을 찾을 수 없습니다: {test_trace_path}")
        print("    예시 데이터를 생성합니다.")
        sample = {"traceID": "51a239b290c1e6e4662092070fd3724b", "spans": []}
        try:
            test_trace_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"    예시 파일 생성됨: {test_trace_path}")
        except Exception as e:
            print(f"    예시 파일 생성 실패: {e}")
        target_trace = sample

    # 4. 유사 트레이스 검색 (Graph RAG)
    similar_traces = retriever.find_similar_traces(
        target_trace_data=target_trace, threshold=0.8, top_k=3
    )

    # 5. GPT 분석 실행 (Chain-of-Thought)
    analysis_result = analyzer.analyze_with_cot(
        target_trace=target_trace, similar_traces=similar_traces
    )

    # 6. 결과 출력
    print("\n" + "=" * 70)
    print("분석 결과")
    print("=" * 70)
    print(f"\n대상 Trace ID: {target_trace.get('traceID')}")
    print(f"참조한 유사 트레이스: {len(similar_traces)}개")
    print("\n" + "-" * 70)
    print(analysis_result)
    print("-" * 70)

    # 7. 결과 저장
    output_file = "trace_analysis_result.json"
    result_data = {
        "target_trace_id": target_trace.get("traceID"),
        "analysis_timestamp": __import__("datetime").datetime.now().isoformat(),
        "similar_traces": similar_traces,
        "analysis": analysis_result,
        "settings": {"similarity_threshold": 0.8, "top_k": 3, "model": "gpt-4o"},
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"\n[+] 결과 저장: {output_file}")

    # 8. 연결 종료
    retriever.close()
    print("\n[+] 분석 완료")


if __name__ == "__main__":
    main()
