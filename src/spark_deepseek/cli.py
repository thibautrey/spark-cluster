import argparse
import asyncio
import json
import os
import sys

import httpx
import uvicorn

from .agent import collect_node
from .capacity import calculate_capacity
from .models import CapacityRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spark-deepseek")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="Run the control plane and dashboard")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", default=8080, type=int)

    agent = commands.add_parser("agent", help="Send local GB10 telemetry to a control plane")
    agent.add_argument("--server", default="http://localhost:8080")
    agent.add_argument("--node-id")
    agent.add_argument("--role", choices=["head", "worker", "cache"], default="worker")
    agent.add_argument("--interval", type=float, default=2)
    agent.add_argument("--once", action="store_true")

    plan = commands.add_parser("plan", help="Estimate cluster memory capacity")
    plan.add_argument("--parameters", type=float, default=671)
    plan.add_argument("--bits", type=float, default=4)
    plan.add_argument("--context", type=int, default=524288)
    plan.add_argument("--concurrency", type=int, default=1)
    return parser


async def run_agent(args: argparse.Namespace) -> None:
    token = os.getenv("SPARK_DEEPSEEK_INGEST_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(timeout=10, headers=headers) as client:
        while True:
            node = collect_node(args.node_id, args.role)
            response = await client.post(
                f"{args.server.rstrip('/')}/api/v1/nodes", json=node.model_dump(mode="json")
            )
            response.raise_for_status()
            print(json.dumps({"node": node.id, "status": "reported"}))
            if args.once:
                return
            await asyncio.sleep(args.interval)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "serve":
        uvicorn.run("spark_deepseek.app:app", host=args.host, port=args.port)
    elif args.command == "agent":
        asyncio.run(run_agent(args))
    elif args.command == "plan":
        result = calculate_capacity(
            CapacityRequest(
                parameter_count_billions=args.parameters,
                bits_per_weight=args.bits,
                context_length=args.context,
                concurrent_sequences=args.concurrency,
            )
        )
        json.dump(result.model_dump(), sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
