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
              {{ message.text }}
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
  </div>
</template>


<script>
import axios from "axios";
import Prism from "prismjs";
import "prismjs/themes/prism.css";
import 'prismjs/components/prism-python';
import "prismjs/plugins/line-numbers/prism-line-numbers.css";
import "prismjs/plugins/line-numbers/prism-line-numbers.min.js";

export default {
  name: "Chat",
  data() {
    return {
      messages: [{id: 0, text:"Hello there, my name is Snippetsage, I can help you find code snippets. What are you looking for?", isBot: true}],
      userMessage: "",
      botIsTyping: false,
      recentSearches: [],
      jsonResponse1: {},
      jsonResponse2: {},
      textValues: [],
      singleTextValue: ""
    };
  },
  methods: {isJson(str) {
    try {
        JSON.parse(str);
    } catch (e) {
        return false;
    }
    return true;
},
    async sendMessage() {
  this.botIsTyping = true;
  this.messages.push({ id: 0, text: this.userMessage, isBot: false });
  try{
  const response = await axios.post("http://localhost:5005/webhooks/rest/webhook", {
    message: this.userMessage,
  });
  this.jsonResponse1 = response.data[0];
  this.jsonResponse2 = response.data[1];
  this.messages.push({ id: 0, text: "I found the following questions that are semantically close to your question!", isBot: true });

    if (Array.isArray(JSON.parse(this.jsonResponse1.text))) {
        console.log(JSON.parse(this.jsonResponse1.text));
        let answer = JSON.parse(this.jsonResponse1.text)
        answer.forEach(element => {
          this.messages.push({ id: 0, text: "Question: " + element._source.question + " with a score of " + element._score, isBot: true });
          this.messages.push({ id: 0, text: element._source.code, isBot: true });

        });
          
          }
          if (typeof this.jsonResponse2.text === "string") {
            this.singleTextValue = this.jsonResponse2.text;
          }
      } catch (error) {
        console.log(error);
      }
  this.messages.push({ id: 0, text: this.singleTextValue, isBot: true });
  this.userMessage = "";
  this.botIsTyping = false;
    },
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

</style>