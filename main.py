from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  
# build a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=75, optimizer='Adam',lr0=0.1, lrf=0.01,imgsz=640,patience=100,batch=16) 
# train the model
