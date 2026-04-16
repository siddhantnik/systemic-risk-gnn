import asyncio
from backend.api.routes import optimize_bailout, BailoutRequest
import backend.services.bailout_optimizer as bo

async def main():
    try:
        req = BailoutRequest(shocked_bank_id="Bank_0000 // Test Bank", budget_millions=500.0)
        res = await optimize_bailout(req)
        print("SUCCESS:", res)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
