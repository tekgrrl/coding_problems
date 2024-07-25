const express = require("express");
const router = express.Router();
const crypto = require("crypto");
const nodemailer = require("nodemailer");
const User = require("../models/User");
const Token = require("../models/Token");
const dotenv = require("dotenv");

const envFile = process.env.NODE_ENV ? `.env.${process.env.NODE_ENV}` : ".env";
dotenv.config({ path: envFile });

// Signup route
router.post("/signup", async (req, res) => {
  try {
    const { email } = req.body;

    if (!email) {
      return res.status(400).json({ error: "Email is required" });
    }

    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: "Email already registered" });
    }

    const newUser = new User({ email });
    await newUser.save();

    const token = crypto.randomBytes(32).toString("hex");
    const newToken = new Token({ token, email });
    await newToken.save();

    console.log(`user = ${process.env.ETHEREAL_USER}`);

    const transporter = nodemailer.createTransport({
      host: "smtp.ethereal.email",
      port: 587,
      auth: {
        user: process.env.ETHEREAL_USER,
        pass: process.env.ETHEREAL_PASS,
      },
    });

    const verificationLink = `http://yourdomain.com/auth/verify?token=${token}`;
    const mailOptions = {
      from: "noreply@yourdomain.com",
      to: email,
      subject: "Verify your email",
      text: `Click this link to verify your email: ${verificationLink}`,
    };

    let info = await transporter.sendMail(mailOptions);

    console.log("Verification email sent: %s", info.messageId);
    console.log("Preview URL: %s", nodemailer.getTestMessageUrl(info));

    res.json({
      message:
        "Signup successful. Please check your email to verify your account.",
      previewUrl: nodemailer.getTestMessageUrl(info),
    });
  } catch (error) {
    console.error("Signup error:", error);
    res.status(500).json({ error: "An error occurred during signup" });
  }
});

// Verify email route
router.get("/verify", async (req, res) => {
  try {
    const { token } = req.query;

    const verificationToken = await Token.findOne({ token });
    if (!verificationToken) {
      return res.status(400).json({ error: "Invalid or expired token" });
    }

    const user = await User.findOne({ email: verificationToken.email });
    if (!user) {
      return res.status(400).json({ error: "User not found" });
    }

    user.verified = true;
    await user.save();

    await Token.deleteOne({ _id: verificationToken._id });

    res.json({ message: "Email verified successfully. You can now sign in." });
  } catch (error) {
    console.error("Verification error:", error);
    res.status(500).json({ error: "An error occurred during verification" });
  }
});

module.exports = router;
