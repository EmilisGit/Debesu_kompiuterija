from ultralytics import YOLO

model = YOLO('yolo11s.pt')

# Run inference on an image
results = model('dog_park.jpg')  # can also use a list of image paths

# Show results (opens a window)
results[0].show()
# Save results
results[0].save('output.jpg')