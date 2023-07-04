import requests
import argparse
import time


def test_predict_endpoint(url: str, filepath: str):
    image_path = filepath

    with open(image_path, "rb") as image_file:
        files = {"file": image_file}

        response = requests.post(url, files=files)

    assert response.status_code == 200


def measure_request_time(url: str, filepath: str):
    num_runs = 100  # Number of requests to send
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        test_predict_endpoint(url, filepath)
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average time per request: {avg_time} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--url", type=str, required=True)
    args = parser.parse_args()
    measure_request_time(args.url, args.file)
