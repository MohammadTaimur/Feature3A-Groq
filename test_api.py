import requests

def test_generate_answers():
    # url = "http://127.0.0.1:8080/generate-answers/"
    url1 = "http://127.0.0.1:8080/check-grammar/"
    question = "What do you do in your free time and why?"

    payload = {
        "question": question
    }

    response = requests.post(url1, data=payload)

    if response.status_code == 200:
        # print("Test Passed!")
        print(response.json())
    else:
        print("Test Failed!")
        print("Status Code:", response.status_code)
        print("Response:", response.text)

if __name__ == "__main__":
    test_generate_answers()
