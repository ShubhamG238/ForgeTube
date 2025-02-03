from ollama import Client
client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)
response = client.chat(model='gemmayt', messages=[
  {
    'role': 'user',
    'content': 'generate me script on tpoic global warming',
  },
])
print(response)