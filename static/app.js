const firebaseConfig = {
    apiKey: "AIzaSyDrFkgnFt43U5x62SWeCnt59h2Q1Et2Q30",
    authDomain: "serverimages-dcad3.firebaseapp.com",
    projectId: "serverimages-dcad3",
    storageBucket: "serverimages-dcad3.firebasestorage.app",
    messagingSenderId: "582482965624",
    appId: "1:582482965624:web:9c43dabf33560a3ba0a98d"
  };
  
  // Inicializa Firebase
firebase.initializeApp(firebaseConfig);
const storage = firebase.storage();

// Función para subir la imagen
function uploadImage() {
  const fileInput = document.getElementById("fileInput");
  const file = fileInput.files[0];
  const status = document.getElementById("status");

  if (!file) {
    status.textContent = "Selecciona una imagen primero.";
    return;
  }

  const storageRef = storage.ref(`imagenes/${file.name}`);
  const uploadTask = storageRef.put(file);

  status.textContent = "Subiendo imagen...";

  uploadTask.on(
    "state_changed",
    null,
    (error) => {
      status.textContent = `Error al subir: ${error}`;
    },
    () => {
      uploadTask.snapshot.ref.getDownloadURL().then((downloadURL) => {
        status.textContent = "✅ Imagen subida con éxito.";
        console.log("URL de la imagen:", downloadURL);
        // Aquí luego enviarás la URL a Colab o backend
      });
    }
  );
}