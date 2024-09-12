# Copilot
For ML

## Setup and Running

Follow these steps to set up and run the project:

1. **Create the .env File**

   In the copilot directory, create a .env file based on the example provided in copilot/.env_example.

2. **Run Docker Compose**

From the root directory of the project, execute the following command:

docker build -t tg_bot
docker run -d -p 8080:80 tg_bot


