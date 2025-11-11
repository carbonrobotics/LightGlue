from lightglue import SuperPoint
import argparse

from lightglue.utils import load_image
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-to-image', type=str, default="/home/raven/data/tmp/lightglue_burst/carbon-images/predict_burst/reapertb1/2024-12-06/de0819f9-cffc-41fb-88b4-2cebbe1ffb6f/predict-burst-record_reapertb1_row1_predict1_2024-12-06T14-17-23-000000Z/predict_burst_reapertb1_row1_predict1_2024-12-06T14-17-21-251000Z.png")
    args = parser.parse_args()
    
    extractor = SuperPoint(resize=1024, max_num_keypoints=2048).eval().cuda() 
    
    
    image = load_image(args.path_to_image).cuda()
    start = time.time()
    feats = extractor.extract(image)
    end = time.time()
    print(f"Time taken to run model 1: {end - start}")
    
    extractor = SuperPoint(resize=4096, max_num_keypoints=2048).eval().cuda() 
    
    
    image = load_image(args.path_to_image).cuda()
    start = time.time()
    feats = extractor.extract(image)
    end = time.time()
    print(f"Time taken to run model 2: {end - start}")

if __name__ == "__main__":
    main()    