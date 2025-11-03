from together import Together

client = Together()

response = client.chat.completions.create(
  model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
  messages=[
    {
      "role": "user",
      "content": "What are some fun things to do in New York?"
    }
  ]
)
print(response.choices[0].message.content)
