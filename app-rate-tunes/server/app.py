import os
import json
from flask import Flask, jsonify, request
from flask_cors import CORS

# POST answer format
PAYLOAD = {
    "musicExperience": "Professional",
    "ratings": {
        "Anthropology_MINGUS": 5,
        "Anthropology_BebopNet": 2,
        "Anthropology_original": 4,
        "Avalon_MINGUS": 5,
        "Avalon_BebopNet": 1,
        "Avalon_original": 3
    },
    "comments": {
        "Anthropology_MINGUS": "",
        "Anthropology_BebopNet": "aweful",
        "Anthropology_original": "great!",
        "Avalon_MINGUS": "",
        "Avalon_BebopNet": "",
        "Avalon_original": ""
    }
}

# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

print('MINGUS server starting')

tunes_file = 'output/TUNES_STATS.json'

if not os.path.exists(tunes_file):
    with open(tunes_file, 'w') as json_file:
        json.dump({}, json_file, indent=4)


@app.route('/', methods=['GET'])
def hello():
    return 'up and running!'


# sanity check route
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')


@app.route('/tunes', methods=['GET', 'POST'])
def all_tunes():
    response_object = {'status': 'success'}
    if request.method == 'POST':
        # open json
        with open(tunes_file) as f:
            TUNES_STATS = json.load(f)
        # get POST request payload
        post_data = request.get_json()
        musicExperience = post_data.get('musicExperience')
        ratings = post_data.get('ratings')
        comments = post_data.get('comments')
        # iterate over rated tunes
        for i in ratings:
            # search for tune in TUNES_STATS json
            for tune in TUNES_STATS:
                if tune['id'] == i:
                    # add tune rating to rating list
                    tune['ratings'].append({
                        'rate': ratings[i],
                        'experience': musicExperience,
                    })
                    tune['comments'].append({
                        'comment': comments[i],
                        'experience': musicExperience,
                    })
        # save updated json file
        with open(tunes_file, 'w') as json_file:
            json.dump(TUNES_STATS, json_file, indent=4)
        response_object['message'] = 'Rating added!'
    else:
        with open('TUNES.json') as f:
            TUNES = json.load(f)
        response_object['tunes'] = TUNES
    return jsonify(response_object)


if __name__ == '__main__':
    app.run()
