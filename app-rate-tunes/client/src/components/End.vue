<template>
<!-- eslint-disable --> 
    <div class="container">
        <div class="row text-center">
            <div class="col-lg-3"></div>
            <div class="col-lg-6">
                <h4>Thank you for partecipating to the quiz!</h4>
            </div>
            <div class="col-lg-3"></div>
        </div>
        <br>
        <div class="row text-center">
            <div class="col-lg-2"></div>
            <div class="col-lg-8">
                <p>You can now submit your answers and discover which 
                    musical phrases were original and which were generated 
                    by a machine learning model.</p>
            </div>
            <div class="col-lg-2"></div>
        </div>
        <br><br>
        <div class="row text-center">
            <div class="col-1 col-lg-2"></div>
                <div class="col-5 col-lg-4">
                  <button type="button" class="btn btn-secondary" @click="backToQuiz">Go back to the quiz</button>
                </div>
                <div class="col-5 col-lg-4">
                  <button type="button" class="btn btn-primary" @click="submitQuiz">Submit and see summary</button>
                </div>
            </div>
            <div class="col-1 col-lg-2"></div>
        </div>
    </div>
</template>

<script>
import Vue from 'vue';
import VueRouter from 'vue-router';
import axios from 'axios';

const server = process.env.NODE_ENV === 'development' ? 'localhost:5000' : 'mingus.tools.eurecom.fr:5552';

Vue.use(VueRouter);

export default {
  data() {
    return {
      ratedTunes: {},
    };
  },
  methods: {
    postRatedTunes(ratedTunes) {
      const path = `http://${server}/tunes`;
      // send ratedTunes to server
      axios.post(path, ratedTunes)
        .then(() => {
          // get answer
        })
        .catch((error) => {
          console.error(error);
        });
    },
    submitQuiz() {
      // check that all tunes have been voted
      let allTunesVoted = true;
      const retrievedRatings = localStorage.getItem('rated-tunes');
      const parsedRatings = JSON.parse(retrievedRatings);
      const keys = Object.keys(parsedRatings);
      for (let i = 0; i < keys.length; i += 1) {
        if (parsedRatings[keys[i]] === 0) {
          allTunesVoted = false;
        }
      }
      if (allTunesVoted) {
        // get ratedTunes from local storage
        // get comments from local storage
        const retrievedComments = localStorage.getItem('comments');
        const parsedComments = JSON.parse(retrievedComments);
        // form the payload for the POST request to server
        this.ratedTunes.musicExperience = localStorage.musicExperience;
        this.ratedTunes.ratings = parsedRatings;
        this.ratedTunes.comments = parsedComments;
        // log the payload to check
        this.postRatedTunes(this.ratedTunes);
        this.$router.push('/summary');
      } else {
        alert("You haven't voted on all tunes, please go back and vote on all tunes");
      }
    },
    backToQuiz() {
      this.$router.push('/tunes');
    },
  },
  created() {
  },
};
</script>
