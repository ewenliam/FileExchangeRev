const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const bodyParser = require('body-parser');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

let userFileSizeLimits = {};
let users = {}; // { username: { password, socketId } }
let onlineUsers = {}; // { socketId: username }

// Create uploads directory if it doesn't exist
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

// Middleware
app.use(bodyParser.json());
app.use(express.static('public'));

// Serve static files from the uploads directory
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Multer storage configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

// Function to get multer configuration based on user limit
const getMulterConfig = (userId) => {
  return multer({
    storage: storage,
    limits: { fileSize: userFileSizeLimits[userId] || 1000000 } // Default to 1MB if no user limit
  }).single('file');
};

// Endpoint to handle file upload with dynamic limits
app.post('/upload/:userId', (req, res) => {
  const userId = req.params.userId;
  const upload = getMulterConfig(userId);

  upload(req, res, (err) => {
    if (err) {
      return res.status(400).send({ error: err.message });
    }
    res.json({ filePath: `/uploads/${req.file.filename}` });
  });
});

// Endpoint for user registration
app.post('/register', (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.status(400).send({ error: 'Username and password are required' });
  }

  if (users[username]) {
    return res.status(400).send({ error: 'Username already exists' });
  }

  users[username] = { password };
  res.json({ message: 'Registration successful' });
});

// Endpoint for user login
app.post('/login', (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.status(400).send({ error: 'Username and password are required' });
  }

  if (!users[username]) {
    return res.status(400).send({ error: 'Username not found' });
  }

  if (users[username].password !== password) {
    return res.status(401).send({ error: 'Invalid credentials' });
  }

  res.json({ message: 'Login successful' });
});

io.on('connection', (socket) => {
  console.log('A user connected: ', socket.id);

  socket.on('disconnect', () => {
    console.log('User disconnected: ', socket.id);
    const username = onlineUsers[socket.id];
    if (username) {
      delete onlineUsers[socket.id];
      delete userFileSizeLimits[socket.id];
      io.emit('userList', Object.values(onlineUsers));
    }
  });

  socket.on('login', (username) => {
    onlineUsers[socket.id] = username;
    io.emit('userList', Object.values(onlineUsers));
  });

  socket.on('sendFile', (data) => {
    console.log('Data received:', data);
    const recipientSocket = Object.keys(onlineUsers).find(key => onlineUsers[key] === data.to);
    if (recipientSocket) {
      io.to(recipientSocket).emit('receiveFile', { fileName: data.fileName, filePath: data.filePath, from: onlineUsers[socket.id] });
    } else {
      console.log('Recipient socket not found:', data.to);
    }
  });
  

  socket.on('checkStatus', (username, callback) => {
    const userSocket = Object.keys(onlineUsers).find(key => onlineUsers[key] === username);
    if (userSocket) {
      callback({ status: 'available' });
    } else {
      callback({ status: 'unavailable' });
    }
  });

  socket.on('setFileSizeLimit', (limit) => {
    userFileSizeLimits[socket.id] = limit;
  });
});
