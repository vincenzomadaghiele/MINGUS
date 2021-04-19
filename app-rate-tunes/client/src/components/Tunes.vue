<template>
<!-- eslint-disable --> 
<div class="row" v-if="dataReady">
<div class="col-lg-3 col-sm-0"></div>
<div class="container col-lg-6">
  <div class="container">
    <div class="text-center">
        <h2>{{ tunes[indcator].title }}</h2>
        <h4>{{ tunes[indcator].author }}</h4>
        <h5>{{ tunes[indcator].sample }}</h5>
    </div>
    <br><br>
    <div class="container">
        <div class="row text-center">
            <div class="col">
              <div :key="indcator">
                <audio style="width: 90%" controls id="myVideo">
                  <source v-bind:src="tunes[indcator].url" type="audio/mp3" :key="indcator" >
                  Your browser does not support the audio element.
                </audio>
              </div>
            </div>
        </div>
        <br><br>
        <div class="row text-center">
          <div class="col">
            <h6>How much do you like this musical phrase?</h6>
          </div>
        </div>
        <br>
        <div class="row text-center">
            <div class="col">
                <ul class="ratings">
                    <li class="star cursor-change" v-bind:class="{ selected: rating == 5}" @click="onVote(5)"></li>
                    <li class="star cursor-change" v-bind:class="{ selected: rating == 4}" @click="onVote(4)"></li>
                    <li class="star cursor-change" v-bind:class="{ selected: rating == 3}" @click="onVote(3)"></li>
                    <li class="star cursor-change" v-bind:class="{ selected: rating == 2}" @click="onVote(2)"></li>
                    <li class="star cursor-change" v-bind:class="{ selected: rating == 1}" @click="onVote(1)"></li>
                </ul>
            </div>
        </div>
        <br><br>
        <div class="row text-center">
          <div class="col">
            <textarea v-model="comment" rows="3" cols="30" placeholder="Why did you vote like this?">
            </textarea>
          </div>
        </div>
        <br><br>
        <div class="row text-center">
          <div class="col-lg-2"></div>
            <div class="col-6 col-lg-4">
                <button type="button" class="btn btn-outline-dark" @click="backToInstruction" v-if="indcator == 0">
                  Back to instructions
                </button>
                <font-awesome-icon icon="chevron-left" class="cursor-change" size="3x" @click="tuneBackward" v-if="indcator > 0"/>
            </div>
            <div class="col-6 col-lg-4">
                <button type="button" class="btn btn-outline-dark" @click="endQuiz" v-if="indcator == 5">
                  End quiz
                </button>
                <font-awesome-icon icon="chevron-right" class="cursor-change" size="3x" @click="tuneForward" v-if="indcator < (tunes.length - 1)"/>
            </div>
            <div class="col-lg-2"></div>
        </div>
        <br><br>
    </div>
  </div>
</div>
<div class="col-lg-3 col-sm-0"></div>
</div>
</template>

<script>
import axios from 'axios';
import Vue from 'vue';
import VueRouter from 'vue-router';

const server = process.env.NODE_ENV === 'development' ? 'http://localhost:5000' : 'https://mingus.tools.eurecom.fr/server';

Vue.use(VueRouter);

export default {
  data() {
    return {
      tunes: [],
      dataReady: false,
      rating: 0,
      indcator: 0,
      comment: null,
    };
  },
  methods: {
    getTunes() {
      const path = `${server}/tunes`;
      const ratings = {};
      axios.get(path)
        .then((res) => {
          if (localStorage.getItem('tunes').length === 0) {
            this.tunes = res.data.tunes;
            localStorage.setItem('tunes', JSON.stringify(this.tunes));
          } else {
            const retrievedTunes = localStorage.getItem('tunes');
            const tunes = JSON.parse(retrievedTunes);
            this.tunes = tunes;
          }
          if (localStorage.getItem('rated-tunes').length === 0) {
            for (let i = 0; i < this.tunes.length; i += 1) {
              ratings[this.tunes[i].id] = 0;
            }
            localStorage.setItem('rated-tunes', JSON.stringify(ratings));
          } else {
            const retrievedRatings = localStorage.getItem('rated-tunes');
            const ratedTunes = JSON.parse(retrievedRatings);
            this.rating = ratedTunes[this.tunes[this.indcator].id];
          }
          if (localStorage.getItem('comments').length === 0) {
            const comments = {};
            for (let i = 0; i < this.tunes.length; i += 1) {
              comments[this.tunes[i].id] = '';
            }
            localStorage.setItem('comments', JSON.stringify(comments));
          } else {
            const retrievedComments = localStorage.getItem('comments');
            const comments = JSON.parse(retrievedComments);
            this.comment = comments[this.tunes[this.indcator].id];
          }
          this.dataReady = true;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
    tuneForward() {
      let comments = {};
      let ratedTunes = {};
      if (this.indcator < (this.tunes.length - 1)) {
        if (localStorage.getItem('comments') !== null) {
          const retrievedObject = localStorage.getItem('comments');
          comments = JSON.parse(retrievedObject);
          comments[this.tunes[this.indcator].id] = this.comment;
          localStorage.setItem('comments', JSON.stringify(comments));
        }
        if (localStorage.getItem('rated-tunes') !== null) {
          const retrievedObject = localStorage.getItem('rated-tunes');
          ratedTunes = JSON.parse(retrievedObject);
          ratedTunes[this.tunes[this.indcator].id] = this.rating;
          localStorage.setItem('rated-tunes', JSON.stringify(ratedTunes));
        }
        this.indcator += 1;
        this.comment = comments[this.tunes[this.indcator].id];
        this.rating = ratedTunes[this.tunes[this.indcator].id];
      }
    },
    tuneBackward() {
      let comments = {};
      let ratedTunes = {};
      if (this.indcator > 0) {
        if (localStorage.getItem('comments') !== null) {
          const retrievedObject = localStorage.getItem('comments');
          comments = JSON.parse(retrievedObject);
          comments[this.tunes[this.indcator].id] = this.comment;
          localStorage.setItem('comments', JSON.stringify(comments));
        }
        if (localStorage.getItem('rated-tunes') !== null) {
          const retrievedObject = localStorage.getItem('rated-tunes');
          ratedTunes = JSON.parse(retrievedObject);
          ratedTunes[this.tunes[this.indcator].id] = this.rating;
          localStorage.setItem('rated-tunes', JSON.stringify(ratedTunes));
        }
        this.indcator -= 1;
        this.comment = comments[this.tunes[this.indcator].id];
        this.rating = ratedTunes[this.tunes[this.indcator].id];
      }
    },
    onVote(startNum) {
      this.rating = startNum;
    },
    backToInstruction() {
      let comments = {};
      let ratedTunes = {};
      if (this.indcator < (this.tunes.length - 1)) {
        if (localStorage.getItem('comments') !== null) {
          const retrievedObject = localStorage.getItem('comments');
          comments = JSON.parse(retrievedObject);
          comments[this.tunes[this.indcator].id] = this.comment;
          localStorage.setItem('comments', JSON.stringify(comments));
        }
        if (localStorage.getItem('rated-tunes') !== null) {
          const retrievedObject = localStorage.getItem('rated-tunes');
          ratedTunes = JSON.parse(retrievedObject);
          ratedTunes[this.tunes[this.indcator].id] = this.rating;
          localStorage.setItem('rated-tunes', JSON.stringify(ratedTunes));
        }
        this.comment = comments[this.tunes[this.indcator].id];
        this.rating = ratedTunes[this.tunes[this.indcator].id];
      }
      this.$router.push('/instructions');
    },
    endQuiz() {
      if (localStorage.getItem('comments') !== null) {
        const retrievedObject = localStorage.getItem('comments');
        const comments = JSON.parse(retrievedObject);
        comments[this.tunes[this.indcator].id] = this.comment;
        localStorage.setItem('comments', JSON.stringify(comments));
      }
      if (localStorage.getItem('rated-tunes') !== null) {
        const retrievedObject = localStorage.getItem('rated-tunes');
        const ratedTunes = JSON.parse(retrievedObject);
        ratedTunes[this.tunes[this.indcator].id] = this.rating;
        localStorage.setItem('rated-tunes', JSON.stringify(ratedTunes));
      }
      this.$router.push('/end');
    },
  },
  created() {
    this.getTunes();
  },
};
</script>

<style>
.cursor-change {
  cursor: pointer;
}

.ratings {
  list-style-type: none;
  margin: 0;
  padding: 0;
  width: 100%;
  direction: rtl;
  text-align: center;
}

.star {
  position: relative;
  line-height: 30px;
  display: inline-block;
  transition: color 0.2s ease;
  color: #ebebeb;
}

.star:before {
  content: '\2605';
  width: 30px;
  height: 30px;
  font-size: 30px;
}

.star:hover,
.star.selected,
.star:hover ~ .star,
.star.selected ~ .star{
  transition: color 0.8s ease;
  color: black;
}
</style>
