<template>
  <div   class="flex h-screen w-screen">
 <Chat/>
 <BaseLineSearch/>
</div>
</template>

<script>
import Chat from "./components/ChatBot.vue";
import BaseLineSearch from "./components/BaseLineSearch.vue"

export default {
  name: "App",
  components: {
    Chat,
    BaseLineSearch
  },
};
</script>