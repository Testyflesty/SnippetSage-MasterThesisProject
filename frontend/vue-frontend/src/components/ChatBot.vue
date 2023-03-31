<template>
  <div class="h-screen w-screen flex">
    <div class="flex-1 flex flex-col bg-gray-900 text-white">
      <div class="flex-1 overflow-y-auto p-6">
        <div v-for="message in messages" :key="message.id" class="flex flex-col mb-4">
          <div v-if="message.isBot" class="flex flex-row items-start mb-2">
            <img src="./bot.png" alt="Bot" class="w-8 h-8 rounded-full mr-2" />
            <div class="bg-gray-800 rounded-lg p-2">
              <pre v-if="isCode(message.text)">
                <code v-html="highlightCode(message.text)"></code>
              </pre>
              <div v-else>
                {{ message.text }}
              </div>
            </div>
          </div>
          <div v-else class="flex flex-row-reverse items-start mb-2">
            <img src="./user.png" alt="User" class="w-8 h-8 rounded-full ml-2" />
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
        <input type="text" v-model="userMessage" placeholder="Type your message here..." class="w-11/12 border border-gray-700 rounded-lg px-4 py-2 mb-4 focus:outline-none focus:border-blue-500 bg-gray-800 text-white" />

        <button type="submit" class="bg-blue-500 w-auto ml-2 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition-colors duration-300">Send</button>
      </form>
    </div>
  </div>
</template>

<script>
import axios from "axios";
import hljs from "highlight.js/lib/core";
import javascript from "highlight.js/lib/languages/javascript";
import python from "highlight.js/lib/languages/python";
import "highlight.js/styles/monokai.css";

hljs.registerLanguage("javascript", javascript);
hljs.registerLanguage("python", python);

export default {
  name: "Chat",
  data() {
    return {
      messages: [],
      userMessage: "",
      botIsTyping: false,
    };
  },
  methods: {
    async sendMessage() {
  this.botIsTyping = true;
  this.messages.push({ id: 0, text: this.userMessage, isBot: false });
      const response = await axios.post("http://localhost:5005/webhooks/rest/webhook", {
        message: this.userMessage,
      });

  console.log(response);

  // Split the message at code snippets and push them separately to the messages array
  response.data.forEach((message) => {
    if (this.isCode(message.text)) {
      const highlightedCode = this.highlightCode(message.text);
      const splitCode = highlightedCode.split("\n");
      splitCode.forEach((code, index) => {
        this.messages.push({
          id: message.id,
          text: code,
          isBot: true,
          isCode: true,
          isFirstLine: index === 0,
          isLastLine: index === splitCode.length - 1,
        });
      });
    } else {
      this.messages.push({
        id: message.id,
        text: message.text,
        isBot: true,
      });
    }
  });

  this.userMessage = "";

  if (response.data[0].text.startsWith("I found")) {
    const searchQuery = response.data[0].text.split(":")[1].trim();
    this.recentSearches.unshift(searchQuery);
  }

  this.botIsTyping = false;
},
    isCode(text) {
      return text.startsWith("```") && text.endsWith("```");
    },
    highlightCode(text) {
      const code = text.slice(3, -3);
      const language = text.slice(3).split("\n")[0].slice(3);
      return hljs.highlight(code, { language }).value;
    },
  },
};
</script>