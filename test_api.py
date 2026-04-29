import requests

jd = "Sr Product Manager AI Cloud SambaNova Systems. Own the AI Cloud strategy that proves hardware dominance. Define how the cloud drives velocity across Enterprise Managed Service and On-Premise product lines. Requirements: 8 plus years product ownership in Cloud Infrastructure AI ML or Compute Platforms. Hardware-Software Fluency with deep understanding of AI accelerators and how hardware architecture dictates inference performance. Inference Market Expertise including hyperscalers and specialized providers. Economic Rigor including tokenomics margin analysis and P&L impact of optimizations. Ability to write technical PRDs and model unit economics."

response = requests.post(
    "http://localhost:8000/benchmark",
    json={"job_description": jd}
)

data = response.json()

for config in data["configs"]:
    print(f"\nConfig: {config['config']}")
    print(f"Latency: {config['latency_ms']}ms")
    print(f"Cost: ${config['cost_usd']}")
    print(f"Quality: {config['quality_score']}/100")
    print(f"Savings: {config['cost_savings_vs_baseline']}")
    print(f"Answer preview: {config['answer'][:200]}")
    print("-" * 50)