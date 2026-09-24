import asyncio
from AX_Voice.AX_Action_Agent import run_ax_action_agent_async

async def main():
    res = await run_ax_action_agent_async("Quiero ver soluciones", "Claro, te voy a abrir la página de soluciones.")
    print("Action Agent Response:", res)

asyncio.run(main())
