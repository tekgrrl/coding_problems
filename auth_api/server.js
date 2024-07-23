const express = require('express');
const bodyParser = require('body-parser');
const nodemailer = require('nodemailer');
const crypto = require('crypto');
const mongoose = require('mongoose');

const app = express();
app.use(bodyParser.json());

// MongoDB connection
mongoose.connect('mongodb://localhost:27017/auth_api', { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('MongoDB connection error:', err));

// User Schema
const userSchema = new mongoose.Schema({
  email: { type: String, unique: true, required: true },
  verified: { type: Boolean, default: false }
});

const User = mongoose.model('User', userSchema);

// Verification Token Schema
const tokenSchema = new mongoose.Schema({
  token: { type: String, required: true },
  email: { type: String, required: true },
  createdAt: { type: Date, default: Date.now, expires: 3600 } // Token expires after 1 hour
});

const Token = mongoose.model('Token', tokenSchema);

// Function to create Ethereal Email transporter
async function createTransporter() {
  let testAccount = await nodemailer.createTestAccount();
  return nodemailer.createTransport({
    host: "smtp.ethereal.email",
    port: 587,
    secure: false, // true for 465, false for other ports
    auth: {
      user: testAccount.user,
      pass: testAccount.pass,
    },
  });
}

// Signup endpoint
app.post('/auth/signup', async (req, res) => {
  try {
    const { email } = req.body;
    
    if (!email) {
      return res.status(400).json({ error: 'Email is required' });
    }

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: 'Email already registered' });
    }

    // Create new user
    const newUser = new User({ email });
    await newUser.save();

    // Generate verification token
    const token = crypto.randomBytes(32).toString('hex');
    const newToken = new Token({ token, email });
    await newToken.save();

    // Create Ethereal Email transporter
    const transporter = await createTransporter();

    // Send verification email
    const verificationLink = `http://yourdomain.com/auth/verify?token=${token}`;
    const mailOptions = {
      from: 'noreply@yourdomain.com',
      to: email,
      subject: 'Verify your email',
      text: `Click this link to verify your email: ${verificationLink}`
    };

    let info = await transporter.sendMail(mailOptions);

    console.log("Verification email sent: %s", info.messageId);
    console.log("Preview URL: %s", nodemailer.getTestMessageUrl(info));

    res.json({ 
      message: 'Signup successful. Please check your email to verify your account.',
      previewUrl: nodemailer.getTestMessageUrl(info)
    });
  } catch (error) {
    console.error('Signup error:', error);
    res.status(500).json({ error: 'An error occurred during signup' });
  }
});

// Verify email endpoint (unchanged)
app.get('/auth/verify', async (req, res) => {
  try {
    const { token } = req.query;

    const verificationToken = await Token.findOne({ token });
    if (!verificationToken) {
      return res.status(400).json({ error: 'Invalid or expired token' });
    }

    const user = await User.findOne({ email: verificationToken.email });
    if (!user) {
      return res.status(400).json({ error: 'User not found' });
    }

    user.verified = true;
    await user.save();

    await Token.deleteOne({ _id: verificationToken._id });

    res.json({ message: 'Email verified successfully. You can now sign in.' });
  } catch (error) {
    console.error('Verification error:', error);
    res.status(500).json({ error: 'An error occurred during verification' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});