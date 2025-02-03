import modal


generate_script = modal.Function.lookup("script_test_app", "script_generator")

prompt = "generate me a script as a youtube content creator talking about the lastest tech news "

with modal.App().run():
    script_data = generate_script.remote(prompt)

output_path = "C:/CRAZY/generated_script.txt"
with open(output_path, "wb") as f:
    f.write(script_data)

print(f"script generated and saved locally at: {output_path}")