from flask import Flask, jsonify, request
import torch
import torchvision.transforms as transforms
from PIL import Image
# imports added by me
import cv2
import numpy as np

import base64
import io

app = Flask(__name__)

# <------------------------------------------------------------>

# Pennylane
import pennylane as qml
from pennylane import numpy as np
import torch.nn as nn

n_qubits = 8                # Number of qubits
# step = 0.0008               # Learning rate
# batch_size = 32             # Number of samples for each training step
# num_epochs = 10              # Number of training epochs
q_depth = 4                 # Depth of the quantum circuit (number of variational layers)
# gamma_lr_scheduler = 0.1    # Learning rate reduction applied every 10 epochs.
# q_delta = 0.01              # Initial spread of random quantum weights
# start_time = time.time()    # Start of the computation timer
# step_size=10  # given in lr_scheduler

dev = qml.device("lightning.qubit", wires=n_qubits)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Variational quantum circuit
# We first define some quantum layers that will compose the quantum circuit.
def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


# Now we define the quantum circuit through the PennyLane [qnode]{.title-ref} decorator .

# The structure is that of a typical variational quantum circuit:

# Embedding layer: All qubits are first initialized in a balanced superposition of up and down states, then they are rotated according to the input parameters (local embedding).
# Variational layers: A sequence of trainable rotation layers and constant entangling layers is applied.
# Measurement layer: For each qubit, the local expectation value of the $Z$ operator is measured. This produces a classical output vector, suitable for additional post-processing.

@qml.qnode(dev, interface="torch")
def quantum_net(q_input_features, q_weights_flat):
    """
    The variational quantum circuit.
    """

    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_input_features)

    # Sequence of trainable variational layers
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    # Expectation values in the Z basis
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)

# Dressed quantum circuit
# We can now define a custom torch.nn.Module representing a dressed quantum circuit.
# This is a concatenation of:
# A classical pre-processing layer (nn.Linear).
# A classical activation function (torch.tanh).
# A constant np.pi/2.0 scaling.
# The previously defined quantum circuit (quantum_net).
# A classical post-processing layer (nn.Linear).
# The input of the module is a batch of vectors with 512 real parameters (features) and the output is a batch of vectors with two real outputs (associated with the two classes of images: ants and bees).

class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the dressed quantum net.
    """

    def _init_(self):
        """
        Definition of the dressed layout.
        """

        super()._init_()
        self.pre_net = nn.Linear(2048, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))
        self.post_net = nn.Linear(n_qubits, 3)

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the dressed quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, n_qubits)
        q_out = q_out.to(device)
        for elem in q_in:
            q_out_elem = torch.hstack(quantum_net(elem, self.q_params)).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        # return the two-dimensional prediction from the postprocessing layer
        return self.post_net(q_out)




# <------------------------------------------------------------->
# # Load your machine learning model here
# # model = torch.load('path_to_your_model')  # Replace with your model loading code

# # Define the data_transform for preprocessing a single image
# single_image_transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

# def preprocess_single_image(image_file):
#     image = Image.open(image_file)
#     image_tensor = single_image_transform(image)
#     return image_tensor
# imports 


# Load the model 
model_path='final_model.pth'
model = torch.load(model_path, map_location=torch.device('cpu'))

# Image preprocessing class
class ImageProcessing:
    def __init__(self, img_height, img_width, no_channels, tol=10, sigmaX=10):

        ''' Initialzation of variables'''

        self.img_height = img_height
        self.img_width = img_width
        self.no_channels = no_channels
        self.tol = tol
        self.sigmaX = sigmaX

    def cropping_2D(self, img, is_cropping = False):

        '''This function is used for Cropping the extra dark part of the GRAY images'''

        mask = img>self.tol
        return img[np.ix_(mask.any(1),mask.any(0))]

    def cropping_3D(self, img, is_cropping = False):

        '''This function is used for Cropping the extra dark part of the RGB images'''

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>self.tol

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # if image is too dark we return the image
            return img
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]  #for channel_1 (R)
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]  #for channel_2 (G)
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]  #for channel_3 (B)
            img = np.stack([img1,img2,img3],axis=-1)
        return img

    def Gaussian_blur(self, img, is_gaussianblur = False):

        '''This function is used for adding Gaussian blur (image smoothing technique) which helps in reducing noise in the image.'''

        img = cv2.addWeighted(img,4,cv2.GaussianBlur(img,(0,0),self.sigmaX),-4,128)
        return img

    def draw_circle(self,img, is_drawcircle = True):

        '''This function is used for drawing a circle from the center of the image.'''

        x = int(self.img_width/2)
        y = int(self.img_height/2)
        r = np.amin((x,y))     # finding radius to draw a circle from the center of the image
        circle_img = np.zeros((self.img_height, self.img_width), np.uint8)
        cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)
        return img

    def image_preprocessing(self, img, is_cropping = True, is_gaussianblur = True):

        """
        This function takes an image -> crops the extra dark part, resizes, draw a circle on it, and finally adds a gaussian blur to the images
        Args : image - (numpy.ndarray) an image which we need to process
           cropping - (boolean) whether to perform cropping of extra part(True by Default) or not(False)
           gaussian_blur - (boolean) whether to apply gaussian blur to an image(True by Default) or not(False)
        Output : (numpy.ndarray) preprocessed image
        """

        if img.ndim == 2:
            img = self.cropping_2D(img, is_cropping)  #calling cropping_2D for a GRAY image
        else:
            img = self.cropping_3D(img, is_cropping)  #calling cropping_3D for a RGB image
        img = cv2.resize(img, (self.img_height, self.img_width))  # resizing the image with specified values
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self.draw_circle(img)  #calling draw_circle
        img = self.Gaussian_blur(img, is_gaussianblur) #calling Gaussian_blur
        return img

# image preprocessing function that takes an PIL image and preprocess it and return the preprocessed image in PIL format
def preprocess_pil_image(input_image_pil,imgh=224,imgw=224,channels=3):
    # Convert PIL image to numpy array
    input_image_np = np.array(input_image_pil)

    # Initialize the ImageProcessing class
    img_processor = ImageProcessing(imgh,imgw,channels)

    # Perform image preprocessing using the ImageProcessing class
    preprocessed_image_np = img_processor.image_preprocessing(input_image_np)

    # Convert preprocessed numpy array back to PIL image
    preprocessed_image_pil = Image.fromarray(preprocessed_image_np)

    return preprocessed_image_pil

# function to run inference on model    
def run_inference(model, img_tensor):
    """
    Run inference using the given model on the provided image tensor.
    
    Args:
        model (torch.nn.Module): The PyTorch model for inference.
        img_tensor (torch.Tensor): The input image tensor.
        
    Returns:
        int: The predicted class label (0, 1, or 2).
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Reshape the image tensor to have batch size 1
    img_tensor = img_tensor.unsqueeze(0)
    print(f"img tensor final shape -> {img_tensor.shape}")
    # Perform inference
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Get the predicted class label
    _, predicted_label = torch.max(outputs, 1)
    print(f"predicted label shape -> {predicted_label.shape}")
    print(f"predicted lab -> {predicted_label}")
    # Convert tensor to integer and return the predicted label
    return int(predicted_label.item())


# defining the tensor transform
single_image_transform = transforms.Compose([
    transforms.ToTensor(),
])


@app.route('/')
def home():
    return "Hello, world!"

@app.route('/predict', methods=['POST'])
def predict():
    # if 'image' in request.files:
    if 'image' in request.form:

        # image_file = request.files['image']
        base64_image = request.form['image']

        # Decode the Base64-encoded image
        image_data = base64.b64decode(base64_image)
        input_image_pil = Image.open(io.BytesIO(image_data))



        # input_image_pil = Image.open(image_file)
        # Preprocess the image using the function
        preprocessed_image_pil = preprocess_pil_image(input_image_pil)

        # You can save or display the preprocessed image as needed
        preprocessed_image_pil.save("preprocessed_image.jpg") # you can comment this 

         

        # converting PIL image to tensor
        img_tensor=single_image_transform(preprocessed_image_pil)
        # making image to run on CPU
        img_tensor = img_tensor.to(torch.device('cpu'))

        #running the inference on model
        pred=run_inference(model,img_tensor)
        # the pred is either 0,1,2
        # defining the classes names for 0,1,2 
        classes=['Normal','Mild','Severe']

        print(f"Prediction : {classes[pred]}")
        
        # Make a prediction using your model
        # Replace this line with actual prediction code
        # prediction = model(preprocessed_image)
        
        # For demonstration purposes, let's assume the prediction result is 'yes'
        result = f"{classes[pred]}"
        # return jsonify({'result': result})
        return result
    else:
        return jsonify({'error': 'No image file found in the request'}), 400

if __name__ == '__main__':
    app.run(debug=True)
