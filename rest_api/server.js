require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const passport = require('passport');

// Connect to MongoDB
mongoose.connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('MongoDB Connected'))
  .catch(err => console.log(err));

const app = express();

// Bodyparser middleware
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

// Passport middleware
app.use(passport.initialize());

// Load User model (do this before we load the passport module)
require('./models/User');

// Passport config
require('./config/passport')(passport);

// Import routes
const users = require('./routes/users');
// Add other resource routes similarly
app.use('/api/users', users);

// Placeholder routes for other resources
// app.use('/api/resource1', require('./routes/api/resource1'));
// Repeat for other resources...

const port = process.env.PORT || 5000;
app.listen(port, () => console.log(`Server up and running on port ${port} !`));
