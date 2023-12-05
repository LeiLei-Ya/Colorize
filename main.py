from colorization_pipline import ImageColorizationPipeline
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
from typing import Union
import numpy as np
import cv2
import io

model_path = 'modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt'
colorizer = ImageColorizationPipeline(model_path=model_path)

app = FastAPI()

@app.post("/api/photo_process/colorize")
async def create_upload_file(file: Union[UploadFile, None] = None):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    color_image = colorizer.process(image)

    # 将处理后的图像编码为JPEG格式
    _, encoded_image = cv2.imencode('.jpg', color_image)
    encoded_image_bytes = encoded_image.tobytes()

    # 创建一个字节流以便发送
    stream = io.BytesIO(encoded_image_bytes)
    
    # 返回图像
    return StreamingResponse(stream, media_type="image/jpeg")

if __name__=="__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
