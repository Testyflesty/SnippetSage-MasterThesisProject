<template>
  <div class="h-screen w-screen flex">
    <div class="flex-1 flex flex-col bg-gray-900 text-white">
      <div class="flex-1 overflow-y-auto p-6">
        <div
          v-for="message in messages"
          :key="message.id"
          class="flex flex-col mb-4"
        >
          <div v-if="message.isBot" class="flex flex-row items-start mb-2">
            <img src="./bot.png" alt="Bot" class="w-8 h-8 rounded-full mr-2" />
            <div class="bg-gray-800 rounded-lg p-2">
              <div v-if="isCode(message.text)">
                <pre>
                <code v-html="highlightCode(message.text)"></code>
              </pre>
              <div class="flex justify-between items-center mt-2">
                <button
                :class="{ 'text-green-500': liked[message.text] }"
                class="mr-1"
                @click="toggleLiked(message.text)"
              >
                <i class="far fa-thumbs-up"></i>
              </button>
              <button
                :class="{ 'text-red-500': disliked[message.text] }"
                class="mr-1"
                @click="toggleDisliked(message.text)"
              >
                <i class="far fa-thumbs-down"></i>
              </button>
                <button @click="thumbsUp(message.text)">👍</button>
                <button @click="thumbsDown(message.text)">👎</button>
              </div>
              
            </div>
              <div v-else>
                {{ message.text }}
              </div>
            </div>
          </div>
          <div v-else class="flex flex-row-reverse items-start mb-2">
            <img
              src="./user.png"
              alt="User"
              class="w-8 h-8 rounded-full ml-2"
            />
            <div class="bg-blue-500 text-white rounded-lg p-2">
              <!-- <Highlighter class="my-highlight" 
                  highlightClassName="highlight"
                  :searchWords="entities"
                  :autoEscape="true"
                  :textToHighlight="message.text"/> -->
                  <p v-if="message.highlightedText === ''">{{ message.text }}</p>
                  <p v-else v-html="message.highlightedText"></p>
                </div>
          </div>
        </div>
        <div v-if="botIsTyping" class="flex flex-row items-start mb-2">
          <div class="bg-gray-800 rounded-lg p-2 mr-2">
            <span class="animate-pulse">Snippetsage is thinking...</span>
          </div>
        </div>
      </div>
      <form @submit.prevent="sendMessage" class="p-6 w-full">
        <input
          type="text"
          v-model="userMessage"
          placeholder="Type your message here..."
          class="w-11/12 border border-gray-700 rounded-lg px-4 py-2 mb-4 focus:outline-none focus:border-blue-500 bg-gray-800 text-white"
        />

        <button
          type="submit"
          class="bg-blue-500 w-auto ml-2 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition-colors duration-300"
        >
          Send
        </button>
      </form>
    </div>
    <div class="bg-gray-800 w-1/4 h-screen">
  <h1 class="text-white font-bold text-xl p-4">Your Searches</h1>
  <div class="px-4">
    <div v-for="(message, index) in usermessages" :key="index" class="py-2 border-b border-gray-700">
      <p class="text-gray-400 mb-1">Message {{ index + 1 }} - Intent: {{ message.intent }}</p>
      <div class="bg-blue-500 text-white rounded-lg p-2">
        <p v-if="message.highlightedText === ''">{{ message.text }}</p>
        <p v-else v-html="message.highlightedText"></p>
      </div>
    </div>
  </div>
</div>
</div>


</template>


<script>
import axios from "axios";
import Prism from "prismjs";
import "prismjs/themes/prism.css";
import 'prismjs/components/prism-python';
import "prismjs/plugins/line-numbers/prism-line-numbers.css";
import "prismjs/plugins/line-numbers/prism-line-numbers.min.js";
import Highlighter from 'vue-highlight-words'

export default {
  name: "Chat",
  components: {
    Highlighter
  },
  data() {
    return {
      messages: [{id: 0, text:"Hello there, my name is Snippetsage, I can help you find code snippets. What are you looking for?", isBot: true, highlightedText:"", intent: ""}],
      userMessage: "",
      botIsTyping: false,
      recentSearches: [],
      jsonResponse1: {},
      jsonResponse2: {},
      
      textValues: [],
      singleTextValue: "",
      namedentities: {},
      formatter: new Intl.NumberFormat('en-US', {
      style: 'percent',
      minimumFractionDigits: 2,
    })
    };
  },
  computed: {
    entities() {
      return Object.keys(this.namedentities);
    },
    entityvalues() {
      return Object.values(this.namedentities);
    },
    usermessages() {
      return this.messages.filter(message => !message.isBot);
    },

  },
  watch: {
  messages: {
    handler: function (newVal, oldVal) {
      for (let i = oldVal.length; i < newVal.length; i++) {
        const message = newVal[i];
        if (!message.isBot && this.namedentities) {
          console.log("highlighting")
          message.highlightedText = this.highlightEntities(
            message.text,
          );
        }
      }
    },
    deep: true,
  },
},
  methods: {isJson(str) {
    try {
        JSON.parse(str);
    } catch (e) {
        return false;
    }
    return true;
},
thumbsUp(text) {
    this.sendFeedback(text, "thumbs_up");
  },
  thumbsDown(text) {
    this.sendFeedback(text, "thumbs_down");
  },
  async sendFeedback(text, feedbackType) {
    try {
      const response = await axios.post("/api/feedback", {
        text: text,
        feedback_type: feedbackType
      });
      console.log(response);
    } catch (error) {
      console.error(error);
    }
  },
highlightEntities(text) {
      const regex = new RegExp(Object.keys(this.namedentities).join('|'), 'gi');
      return text.replace(regex, match => {
        const entity = this.namedentities[match];
        return `<span class="entity-${entity}">${match}<span class="label">[${entity}]</span></span>`;
      });
    },


    async sendMessage() {
    this.botIsTyping = true;
    this.messages.push({ id: 0, text: this.userMessage, isBot: false , highlightedText:""});
    
    try{
      const response = await axios.post("http://localhost:5005/webhooks/rest/webhook", {
      message: this.userMessage,
    });



    console.log(response.data[0])
    let responseData = null;
    try{
      responseData = JSON.parse(response.data[0].text)
    } catch{
      responseData = null;
    }

    if(responseData != null){
    const results = responseData.results

    const intent = responseData.intent
    const entities = responseData.entities

    for (let i = 0; i < entities.length; i++) {
      const entity = entities[i].entity;
      const name = entities[i].value;
      this.namedentities[name]= entity;

    }

    const messageToUpdate = this.messages[this.messages.length - 1];
    console.log(messageToUpdate)
    if (messageToUpdate) {
      messageToUpdate.highlightedText = this.highlightEntities(messageToUpdate.text);
      messageToUpdate.intent = intent;
    }

    
    // loop through the results array to get the questions and code snippets
    for (let i = 0; i < results.length; i++) {
      const question = results[i]._source.question
      const codeSnippet = results[i]._source.code
      const score = this.formatter.format(results[i]._score)

      this.messages.push({ id: i, text: "Question: " + question + " with a score of " + score, isBot: true , highlightedText:""});
      this.messages.push({ id: i, text: codeSnippet, isBot: true ,highlightedText:""});

    }



    const keyValuePairs = Object.entries(this.namedentities).map(([key, value]) => `${key}: ${value}`).join(', ');

    this.messages.push({ id: 0, text: "I classified your question with the following intent: " + intent + " And found these entities: " + keyValuePairs, isBot: true });

      } else {
        this.messages.push({ id: 0, text: response.data[0].text, isBot: true });
      }
          } catch (error) {
        console.log(error);
      }
    
  this.namedentities = {};
  this.userMessage = "";
  this.botIsTyping = false;
    }
    ,
    isCode(text) {
      return text.startsWith("```") && text.endsWith("```");
    },
    highlightCode(text) {
      let code = text.slice(3, -3);
      const highlightedCode = Prism.highlight(code, Prism.languages.python, 'python');
      const codeLines = highlightedCode.split('\n').map((line, i, arr) => {
    if (i === arr.length - 1 && line === '') return '';
    return `<span class="line-number">${i + 1}</span>${line}`;
  }).join('\n');
      return codeLines;
    },

  },
};
</script>
<style>
.my-2:last-child {
  margin-bottom: 0;
}
.line-number {
  display: inline-block;
  width: 2em;
  user-select: none;
  opacity: 0.5;
  text-align: right;
  margin-right: 0.5em;
  color: #999;
}
pre {
  background-color: #2d2d2d;
  color: #fff;
  padding: 0.5em;
  margin: 0.5em 0;
  overflow: auto;
  border-radius: 0.3em;
  white-space: pre-wrap;
  font-size: 0;
}

code {
  font-size: 16px;

  display: block;
  padding: 1em;
}

.label {
  font-size: 0.8em;
  margin-left: 0.2em;
  opacity: 0.5;
  color: #1a202c;
}

.entities {
    line-height: 2;
}
.entity-proglanguage {
  position: relative;
  display: inline-block;
  padding: 0.1em 0.5em;
  border-radius: 0.5em;
  background-color: #fff5eb;
  color: #dd6b20;
}

.entity-query {
  position: relative;
  display: inline-block;
  padding: 0.1em 0.5em;
  border-radius: 0.5em;
  background-color: #fff5eb;
  color: #9538a1;
}

.entity-data_structure {
  position: relative;
  display: inline-block;
  padding: 0.1em 0.5em;
  border-radius: 0.5em;
  background-color: #eaf2f8;
  color: #1a202c;
}

.entity-operation {
  position: relative;
  display: inline-block;
  padding: 0.1em 0.5em;
  border-radius: 0.5em;
  background-color: #f0fff4;
  color: #38a169;
}


.entity-package {
  position: relative;
  display: inline-block;
  padding: 0.1em 0.5em;
  border-radius: 0.5em;
  background-color: #ebf8ff;
  color: #3182ce;
}

.entity-response {
  position: relative;
  display: inline-block;
  padding: 0.1em 0.5em;
  border-radius: 0.5em;
  background-color: #fff5eb;
  color: #FFB8D1;
}


</style>