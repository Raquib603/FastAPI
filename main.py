import os
print("Current working directory:", os.getcwd())

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from vix_analysis import run_vix_analysis

app = FastAPI()

@app.get("/")
def home():
    return {"message": "VIX-Forex Sensitivity API is running!"}

@app.get("/vix-analysis")
def vix_analysis():
    result, plot_path = run_vix_analysis()
    return {"data": result, "plot_path": plot_path}

@app.get("/vix-plot")
def vix_plot():
    _, plot_path = run_vix_analysis()
    return FileResponse(plot_path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
