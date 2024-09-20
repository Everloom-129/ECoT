# 1. Initialize the client with your API token.
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk import DetectionTask
from dds_cloudapi_sdk import TextPrompt
from dds_cloudapi_sdk import DetectionModel
from dds_cloudapi_sdk import DetectionTarget




token = "df9039b5ed73edb8f4fb0c23a7c1e2d2"
config = Config(token)
client = Client(config)

# 2. Upload local image to the server and get the URL.
infer_image_url = client.upload_file("./test_obs.png")
# infer_image_url = client.upload_file("path/to/infer/image.jpg")  # you can also upload local file for processing


task = DetectionTask(
    image_url=infer_image_url,
    prompts=[TextPrompt(text="red watermelon. spoon")],
    targets=[DetectionTarget.Mask, DetectionTarget.BBox],  # detect both bbox and mask
    model=DetectionModel.GDino1_5_Pro,  # detect with GroundingDino-1.5-Pro model
)

# 4. Run the task and get the result.
client.run_task(task)

# 5. Parse the result.
from dds_cloudapi_sdk.tasks.ivp import TaskResult

result: TaskResult = task.result
print(result.mask_url)

# 6. save image
from PIL import Image, ImageDraw

bbox_image = Image.open('./test_obs.png')

draw = ImageDraw.Draw(bbox_image)

objects = result.objects  # the list of detected objects
for idx, obj in enumerate(objects):
    print(f"Object {idx}:")
    print(f"Score: {obj.score}")  
    print(f"Category: {obj.category}") 
    print(f"BBox: {obj.bbox}")  
    # print(obj.mask.counts)  # RLE compressed to string, ]o`f08fa14M3L2O2M2O1O1O1O1N2O1N2O1N2N3M2O3L3M3N2M2N3N1N2O...

    # convert the RLE format to RGBA image
    mask_image = task.rle2rgba(obj.mask)
    print(mask_image.size)  # (1600, 1170)

    # save the image to file
    mask_image.save(f"data/mask_{obj.category}_{idx}.png")

    draw.rectangle(obj.bbox, outline="red", width=3)
    bbox_image.save(f"data/bbox_{obj.category}_{idx}.png")
    break