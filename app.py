import os
import json
from pathlib import Path
from flask import Flask, jsonify, abort
from dotenv import load_dotenv

from trace_analyzer_lib import Neo4jProcessor, TraceAnalyzer

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_DIR = Path(__file__).resolve().parent / "data"

app = Flask(__name__)
neo4j_processor = Neo4jProcessor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
analyzer = TraceAnalyzer(neo4j_processor.driver, OPENAI_API_KEY)

# api 엔드포인트 / trace_id로 호출
@app.route('/analyze/<string:trace_id>', methods=['GET'])
def analyze_trace_endpoint(trace_id):
    print(f"\n[API 요청] /analyze/{trace_id}")
    
    # 1. DB에서 원본 트레이스 데이터 가져오기
    target_trace = neo4j_processor.get_trace_data_by_id(trace_id)
    if not target_trace:
        abort(404, description=f"Trace ID '{trace_id}' not found.")

    # 2. 유사 트레이스 검색
    similar_traces = analyzer.find_similar_traces(target_trace)
    
    # 3. llm을 활용한 최종 분석
    analysis_result = analyzer.analyze_with_cot(target_trace, similar_traces)
    
    return jsonify({
        "target_trace_id": trace_id,
        "llm_analysis_report": analysis_result
    })

# --- 서버 실행 ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
