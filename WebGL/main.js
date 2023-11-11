import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';  
import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';
import WebGL from 'three/addons/capabilities/WebGL.js';


// Import data from files



// Setup button controls



// Create the scene and the renderer
const scene = new THREE.Scene();
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerHeight, window.innerHeight / 1.5);
renderer.setClearColor(0x0000ff, 0.4);

// Attach the renderer to the screen.
const pageAnimationContainer = document.getElementById("animation_container");
pageAnimationContainer.appendChild(renderer.domElement);

// Create a camera that gives us a field of view to the scene. I don't know what all these ratios and variables do and nor do I want to know so long as it works.
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(400, 500, 400);
camera.lookAt(0, 0, 0); // Makes sure we are looking at the right spot.

// Add some orbital controls because why not.
const controls = new OrbitControls(camera, renderer.domElement);
controls.minDistance = 100;
controls.maxDistance = 1000;

// Lights so we can see anything on the screen.
const pointLight1 = new THREE.PointLight(0xffffff);
pointLight1.position.set(1000,1000,1000);
const pointLight2 = new THREE.PointLight(0xffffff);
pointLight2.position.set(-1000, 1000, -1000)
const pointLight3 = new THREE.PointLight(0xffffff);
pointLight3.position.set(-1000, 1000, 1000);
const pointLight4 = new THREE.PointLight(0xffffff);
pointLight4.position.set(1000, 1000, -1000);
// Add the lights to the scene
scene.add(pointLight1);
scene.add(pointLight2);
scene.add(pointLight3);
scene.add(pointLight4);


// Define the 3D Objects for the pendulum
const tower = new THREE.Object3D();
const pendelArm1 = new THREE.Object3D();
const pendelArm2 = new THREE.Object3D();
const pendelBall = new THREE.Object3D();

// Load .obj model files and corresponding .mtl texture files.
const mtl_loader = new MTLLoader();

mtl_loader.load("./model/tower.mtl", function (materials) {
    materials.preload();
    const obj_loader = new OBJLoader();
    obj_loader.setMaterials(materials);
    obj_loader.load("./model/tower.obj", function (object) {
        base.add(object);
    });
});

mtl_loader.load("./model/pendelArm1.mtl", function (materials) {
    materials.preload();
    const obj_loader = new OBJLoader();
    obj_loader.setMaterials(materials);
    obj_loader.load("./model/pendelArm1.obj", function (object) {
        body1.add(object);
    });
});

mtl_loader.load("./model/pendelArm2.mtl", function (materials) {
    materials.preload();
    const obj_loader = new OBJLoader();
    obj_loader.setMaterials(materials);
    obj_loader.load("./model/pendelArm2.obj", function (object) {
        body2.add(object);
    })
});

mtl_loader.load("./model/pendelBall.mtl", function (materials) {
    materials.preload();
    const obj_loader = new OBJLoader();
    obj_loader.setMaterials(materials);
    obj_loader.load("./model/pendelBall.obj", function (object) {
        body3.add(object);
    });
});

// Attach the parts together to form the complete model.
tower.add(pendelArm1);
pendelArm1.add(pendelArm2);
pendelArm2.add(pendelBall);

// Make a check that WebGL is available to be used on the device seeing the site. 
if (WebGL.isWebGLAvailable()) {
    animate();
} else {
    alert("Error in loading WebGL, cannot proceed to animate the model.")
}

function animate()
{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);

    // TODO: Write code for animating with proper timesteps between each datapoint.
}

function reset_model ()
{
    pendelArm1.rotation.x = 0;
    pendelArm2.rotation.x = 0;
}



// Present graph images corresponding to the animation
