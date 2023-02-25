import subprocess
import json
text_dict = {"title": "Wget POST","text": "hello","userId":1}
process = subprocess.run(['wget', '--method=post', '-O-', '-q', '--body-data=' + json.dumps(text_dict), 'localhost:5005/model/parse'],
                        stdout=subprocess.PIPE,
                        universal_newlines=True, stderr=subprocess.PIPE, text=True)
print(eval(process.stdout))
# print(process.stdout)

# wget --method=post -O- -q --body-data='{"title": "Wget POST","text": "hello","userId":1}' localhost:5005/model/parse


# ['wget --method=post -O- -q', --body-data=json.dumps(text_dict), 'localhost:5005/model/parse']

