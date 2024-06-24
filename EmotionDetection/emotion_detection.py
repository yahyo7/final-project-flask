import requests
import json

def emotion_detector(text_to_analyse):
    if not text_to_analyse.strip():  # Check for empty or blank input
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = {"raw_document": {"text": text_to_analyse}}
    
    try:
        response = requests.post(url, json=myobj, headers=header)
        if response.status_code == 400:
            return {
                'anger': None,
                'disgust': None,
                'fear': None,
                'joy': None,
                'sadness': None,
                'dominant_emotion': None
            }
        
        response.raise_for_status()  # Raise an HTTPError for bad responses
        formatted_response = response.json()  # Use response.json() for JSON response

        emotion = formatted_response['emotionPredictions'][0]["emotion"]
        max_emotion = max(emotion.items(), key=lambda x: x[1])
        anger_score = emotion['anger']
        disgust_score = emotion['disgust']
        fear_score = emotion['fear']
        joy_score = emotion['joy']
        sadness_score = emotion['sadness']
        dominant_emotion = max_emotion[0]
        return {
            'anger': anger_score,
            'disgust': disgust_score,
            'fear': fear_score,
            'joy': joy_score,
            'sadness': sadness_score,
            'dominant_emotion': dominant_emotion
        }
    except requests.exceptions.RequestException as e:
        print(f"HTTP request failed: {e}")
        return {"error": "HTTP request failed"}
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return {"error": "JSON decode error"}
    except KeyError as e:
        print(f"Missing key in JSON response: {e}")
        return {"error": f"Missing key in JSON response: {e}"}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"error": "An unexpected error occurred"}
