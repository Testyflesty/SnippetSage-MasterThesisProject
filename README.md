# SnippetSage - A Neural Code Search Engine with Intent Modelling Chat Bot

> "Intent modeling enables us to bridge the gap between what users search for and what they really want." - Anne Aula, Director of User Research at Google

This project is a part of my Master's thesis research at Utrecht University, and it involves the development of a neural code search engine that uses intent modelling to improve the accuracy of code search results. Additionally, the project includes a chat bot that uses the same intent modelling technique to help users find relevant code snippets or ask questions about coding concepts. It currently uses CodeBert to create the vector embeddings for the code snippets, which could be replaced by CODEX or other future algorithms. The study is part of the SearchSeco research group at Utrecht University, and therefore the system is compatable with custom and/or private software ecosystems.


Read the associated paper here: 



## Table of Contents

| Section              | Description                              
| ---------------------| ----------------------------------------
|📝 [Project Overview](#project-overview)    | Overview of the project goals and scope   
|📦 [Project Dependencies](#project-dependencies) | List of required dependencies            
|🚀 [Installation](#installation)         | Instructions for installing the project 
|🕹️ [Usage](#usage)                | Instructions for using the project      
|🕹️ [Examples](#examples)                | Example screenshots of using the bot      
|🤝 [Contributing](#contributing)         | Guidelines for contributing to the project 
|📜 [License](#license)              | Information about the project's license  

## Project Overview

The aim of this project is to develop a neural code search engine that uses intent modelling to improve the accuracy of search results. Intent modelling involves identifying the user's intent behind a search query and using that intent to improve the relevance of the search results. In addition, the project includes a chat bot that uses the same intent modelling technique to help users find relevant code snippets.

The code search engine is built using a neural network architecture that uses embeddings to represent code snippets and search queries in a high-dimensional space. The search engine uses these embeddings to find the most relevant code snippets for a given search query.

The chat bot is built using a natural language processing (NLP) framework, which allows it to understand user queries and provide relevant responses. The chat bot uses the same neural network architecture as the code search engine to find relevant code snippets based on the user's intent. The intent modelling uses rules along with the DIETclassifier introduced by RASA, one of the project dependencies.

## Project Dependencies

The project requires the following main dependencies:

- Python 3.7
- TensorFlow 2.8.4
- ElasticSearch 7.17.8
- Rasa 3.4.1

For more dependencies check the requirements.txt file in the backend folder. This can be installed using <br/>
`pip install requirements.txt`


## Installation
TODO update the elasticsearch backend commands
To install the project, follow these steps:

1. Clone the repository:  
`git clone https://github.com/Testyflesty/SnippetSage-MasterThesisProject/`

2. CD into the backend directory:  
`cd backend`

3. Run the Rasa action server and the rasa socketIO API:  
`rasa run actions --cors "*" --debug`  
`rasa run -m models --enable-api --cors "*" --debug`
  
4. Use any frontend or serve the index.html file from the frontend folder


## Usage

## Examples

![OLDIntentify](screenshots/OLDIntentify.png?raw=true "Old Intentify Prototype")

![Chitchat](screenshots/Chitchat.png?raw=true "SnippetSage ChitChat")
![Question1](screenshots/Question1.png?raw=true "SnippetSage Asking a code question")
![Question2](screenshots/Question2.png?raw=true "SnippetSage Asking a code question")
![Question3](screenshots/Question3.png?raw=true "SnippetSage Asking a code question")
![Feedback1](screenshots/Feedback1.png?raw=true "SnippetSage Giving feedback")
![Feedback2](screenshots/Feedback2.png?raw=true "SnippetSage Giving feedback")


## Contributing

Contributions to the project are welcome. To contribute, follow these steps:

1. Fork the repository.  
2. Create a new branch for your changes:  

`git checkout -b my-new-branch`  


3. Make your changes and commit them:  

`git commit -am 'Add some feature'`  


4. Push your changes to your fork:  

`git push origin my-new-branch`  


5. Create a pull request on GitHub.  

## License
This project is licensed under the GNU3 License - see the [LICENSE.md](LICENSE.md) file for details.
