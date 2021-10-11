const { app, BrowserWindow, protocol } = require('electron');
const path = require('path');
const url = require('url');

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow = null;

function createWindow() {
  const WEB_FOLDER = 'docs';
  const PROTOCOL = 'file';

  protocol.interceptFileProtocol(PROTOCOL, (request, callback) => {
    // // Strip protocol
    let url = request.url.substr(PROTOCOL.length + 1);

    // Build complete path for node require function
    url = path.join(__dirname, WEB_FOLDER, url);

    // Replace backslashes by forward slashes (windows)
    url = path.normalize(url);
    callback({ path: url });
  });
  // Create the browser window.
  mainWindow = new BrowserWindow({
    show: false,
    webPreferences: {
      webgl: true,
      nodeIntegration: false
    }
  });

  mainWindow.maximize();
  mainWindow.show();
  // and load the index.html of the app.
  mainWindow.loadURL(url.format({
    pathname: 'index.html',
    protocol: PROTOCOL + ':',
    slashes: true
  }));

  // Open the DevTools.
  // mainWindow.webContents.openDevTools()

  // Emitted when the window is closed.
  mainWindow.on('closed', () => {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null;
  });
}

app.on('ready', () => {
  createWindow();
});

//create the application window if the window variable is null
app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

//quit the app once closed
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    process.exit();
    app.quit();
  }
});