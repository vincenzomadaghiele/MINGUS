<template>
<!-- eslint-disable --> 
    <div class="container">
        <div class="row text-center">
            <div class="col-lg-2"></div>
            <div class="col-lg-8">
                <h2>Quiz instructions</h2>
            </div>
            <div class="col-lg-2"></div>
        </div>
        <br>
        <div class="row text-center">
            <div class="col-lg-3"></div>
            <div class="col-lg-6">
                <p>This quiz was build to evaluate <b>MINGUS</b>, a machine learning model
                    for generating jazz improvisations.</p>
                <p>You will listen to musical phrases played on five famous jazz tunes. 
                    Your task will be to evaluate how much you like each musical phrase, 
                    giving it a score from 1 (dislike) to 5 stars (really liked it).</p>
                <p>If you like, you can give us a textual feedback on your choice! 
                    This help us in better evaluating our work.</p>
                <p>You will listen to 3 different musical phrases for each tune, 
                    some of them generated by a machine and some of them from a transcription of the original solos. 
                    At the end of the quiz you will know which are which!</p>
                <h6>So, let's start!</h6>
            </div>
            <div class="col-lg-3"></div>
        </div>
        <br>
        <div class="row text-center">
            <div class="col">
                <button type="button" class="btn btn-primary" @click="startQuiz">Start the quiz</button>
            </div>
        </div>
        <br><br>
    </div>
</template>

<script>
import Vue from 'vue';
import VueRouter from 'vue-router';

Vue.use(VueRouter);

export default {
  data() {
    return {
    };
  },
  methods: {
    startQuiz() {
      this.$router.push('/tunes');
    },
  },
  created() {
  },
};
</script>

<style>
</style>
