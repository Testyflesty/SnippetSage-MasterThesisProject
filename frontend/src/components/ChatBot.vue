<template>
  <div class="h-screen w-screen flex">
    <div class="flex-1 flex flex-col bg-gray-900 text-white">
      <div class="flex-1 overflow-y-auto p-6">
        <div
          v-for="message in filteredmessages"
          :key="message.id"
          class="flex flex-col mb-4"
        >
          <div v-if="message.isBot" class="flex flex-row items-start mb-2">
            <img src="./bot.png" alt="Bot" class="w-8 h-8 rounded-full mr-2" />
            <div :class="{ 'bg-green-200': message.liked, 'bg-red-200': message.disliked }" class="bg-gray-800 rounded-lg p-2">              
              <div v-if="isCode(message.text)">
                <pre>
                <code v-html="highlightCode(message.text)"></code>
              </pre>
              <div class="flex justify-between items-center mt-2">
              <button
                v-if="!message.like && !message.dislike"
                class="px-3 py-1 text-green-600  rounded mr-2"
                @click="likeMessage(message)"
              >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6">
  <path stroke-linecap="round" stroke-linejoin="round" d="M6.633 10.5c.806 0 1.533-.446 2.031-1.08a9.041 9.041 0 012.861-2.4c.723-.384 1.35-.956 1.653-1.715a4.498 4.498 0 00.322-1.672V3a.75.75 0 01.75-.75A2.25 2.25 0 0116.5 4.5c0 1.152-.26 2.243-.723 3.218-.266.558.107 1.282.725 1.282h3.126c1.026 0 1.945.694 2.054 1.715.045.422.068.85.068 1.285a11.95 11.95 0 01-2.649 7.521c-.388.482-.987.729-1.605.729H13.48c-.483 0-.964-.078-1.423-.23l-3.114-1.04a4.501 4.501 0 00-1.423-.23H5.904M14.25 9h2.25M5.904 18.75c.083.205.173.405.27.602.197.4-.078.898-.523.898h-.908c-.889 0-1.713-.518-1.972-1.368a12 12 0 01-.521-3.507c0-1.553.295-3.036.831-4.398C3.387 10.203 4.167 9.75 5 9.75h1.053c.472 0 .745.556.5.96a8.958 8.958 0 00-1.302 4.665c0 1.194.232 2.333.654 3.375z" />
</svg>
              </button>
              <button
                v-if="!message.like && !message.dislike"
                class="px-3 py-1 text-red-600 rounded"
                @click="dislikeMessage(message);"
              >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6">
  <path stroke-linecap="round" stroke-linejoin="round" d="M7.5 15h2.25m8.024-9.75c.011.05.028.1.052.148.591 1.2.924 2.55.924 3.977a8.96 8.96 0 01-.999 4.125m.023-8.25c-.076-.365.183-.75.575-.75h.908c.889 0 1.713.518 1.972 1.368.339 1.11.521 2.287.521 3.507 0 1.553-.295 3.036-.831 4.398C20.613 14.547 19.833 15 19 15h-1.053c-.472 0-.745-.556-.5-.96a8.95 8.95 0 00.303-.54m.023-8.25H16.48a4.5 4.5 0 01-1.423-.23l-3.114-1.04a4.5 4.5 0 00-1.423-.23H6.504c-.618 0-1.217.247-1.605.729A11.95 11.95 0 002.25 12c0 .434.023.863.068 1.285C2.427 14.306 3.346 15 4.372 15h3.126c.618 0 .991.724.725 1.282A7.471 7.471 0 007.5 19.5a2.25 2.25 0 002.25 2.25.75.75 0 00.75-.75v-.633c0-.573.11-1.14.322-1.672.304-.76.93-1.33 1.653-1.715a9.04 9.04 0 002.86-2.4c.498-.634 1.226-1.08 2.032-1.08h.384" />
</svg>
              </button>
              <TransitionRoot as="template" :show="open">
      <Dialog as="div" class="relative z-10" @close="open = false">
        <TransitionChild as="template" enter="ease-out duration-300" enter-from="opacity-0" enter-to="opacity-100" leave="ease-in duration-200" leave-from="opacity-100" leave-to="opacity-0">
          <div class="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" />
        </TransitionChild>
  
        <div class="fixed inset-0 z-10 overflow-y-auto">
          <div class="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
            <TransitionChild as="template" enter="ease-out duration-300" enter-from="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95" enter-to="opacity-100 translate-y-0 sm:scale-100" leave="ease-in duration-200" leave-from="opacity-100 translate-y-0 sm:scale-100" leave-to="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95">
              <DialogPanel class="relative transform overflow-hidden rounded-lg bg-white px-4 pb-4 pt-5 text-left shadow-xl transition-all sm:my-8 sm:w-full sm:max-w-lg sm:p-6">
                <div class="sm:flex sm:items-start">
                  <div class="mx-auto flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-red-100 sm:mx-0 sm:h-10 sm:w-10">
                    <ExclamationTriangleIcon class="h-6 w-6 text-red-600" aria-hidden="true" />
                  </div>
                  <div class="mt-3 text-center sm:ml-4 sm:mt-0 sm:text-left">
                    <DialogTitle as="h3" class="text-base font-semibold leading-6 text-gray-900">Remove the code snippet?</DialogTitle>
                    <div class="mt-2">
                      <p class="text-sm text-gray-500">I'm sorry you did not like the result I gave you. Your search results will be personilized and the algorithm will be trained to provide you with better results.</p>
                    </div>
                  </div>
                </div>
                <div class="mt-5 sm:mt-4 sm:flex sm:flex-row-reverse">
                  <button type="button" class="inline-flex w-full justify-center rounded-md bg-red-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-red-500 sm:ml-3 sm:w-auto" @click="this.removeCodeSnippet(this.modalMessage);open = false;">Remove!</button>
                  <button type="button" class="mt-3 inline-flex w-full justify-center rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50 sm:mt-0 sm:w-auto" @click="open = false" ref="cancelButtonRef">Cancel</button>
                </div>
              </DialogPanel>
            </TransitionChild>
          </div>
        </div>
      </Dialog>
    </TransitionRoot>
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
    <div class="bg-gray-800 w-1/4 h-screen overflow-y-auto">
  <h1 class="text-white font-bold text-xl p-4">Your Searches</h1>
  <div class="px-4">
    <div v-for="(message, index) in usermessages" :key="index" class="py-2 border-b border-gray-700">
      <p class="text-gray-400 mb-1">Message {{ index + 1 }} - Intent: {{ message.intent }}</p>
      <div class="bg-blue-500 text-white rounded-lg p-2">
        <p v-if="message.highlightedText === ''">{{ message.text }}</p>
        <p v-else v-html="message.highlightedText"></p>
      </div>
    </div>
    <button @click="saveMessagesToJson()" class="bg-green-500 w-auto text-white p-2 rounded-lg hover:bg-green-600 transition-colors duration-300">Save Searches</button>
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
import { Dialog, DialogPanel, DialogTitle, TransitionChild, TransitionRoot } from '@headlessui/vue'
import { ExclamationTriangleIcon } from '@heroicons/vue/24/outline'


export default {
  name: "Chat",
  components: {
    Highlighter,
    Dialog,
    DialogPanel,
    DialogTitle,
    TransitionChild,
    TransitionRoot,
    ExclamationTriangleIcon
  },
  data() {
    return {
      messages: [{id: 0, text:"Hello there, my name is Snippetsage, I can help you find code snippets. What are you looking for?", isBot: true, highlightedText:"", intent: "", liked: false, disliked: false}],
      userMessage: "",
      botIsTyping: false,
      recentSearches: [],
      jsonResponse1: {},
      jsonResponse2: {},
      open: false,
      modalMessage: {},
      textValues: [],
      singleTextValue: "",
      modalIsActive: false,
      modalMessage: '',
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
    filteredmessages() {
      return this.messages.filter(message => !message.isRemoved);
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


likeMessage(message) {
      message.liked = true;
      message.disliked = false;
      // this.messages = this.messages.filter((m) => m !== message);
    },
    dislikeMessage(message) {
      message.liked = false;
      message.disliked = true;
      // this.messages = this.messages.filter((m) => m !== message);
      setTimeout(() => {
        this.open = true;
        this.modalMessage = message;
      }, 1000);
    },
    removeCodeSnippet(message) {
      this.messages[message.id].isRemoved = true;
      this.messages[message.id+1].isRemoved = true;

    },

    async saveMessagesToJson() {
      const timestamp = new Date().toISOString();
      const messages = this.messages;
      const data = {
        timestamp,
        messages
      };

      try {
        const response = await axios.post('http://localhost:8080/save-messages', data);
        console.log('Messages saved to JSON file!');
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
    this.messages.push({ id: this.messages.length -1 , text: this.userMessage, isBot: false , highlightedText:""});
    
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

      this.messages.push({ id: this.messages.length -1, text: "Question: " + question + " with a score of " + score, isBot: true , highlightedText:"", score: score});
      this.messages.push({ id: this.messages.length -1, text: codeSnippet, isBot: true ,highlightedText:"", score: score});

    }



    const keyValuePairs = Object.entries(this.namedentities).map(([key, value]) => `${key}: ${value}`).join(', ');

    this.messages.push({ id: this.messages.length -1, text: "I classified your question with the following intent: " + intent + " And found these entities: " + keyValuePairs, isBot: true });

      } else {
        this.messages.push({ id: this.messages.length -1, text: response.data[0].text, isBot: true });
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
      if(text != undefined)
        return text.startsWith("```") && text.endsWith("```");
      else
        return '';
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