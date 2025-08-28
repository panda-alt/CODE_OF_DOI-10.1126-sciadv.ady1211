# CODE_OF_DOI-10.1126-sciadv.ady1211
Code for the article “Phase transitions perception in non-reciprocal mechanical metamaterials through electromagnetic resonance“

# Metamaterial Dynamics and Electromagnetic Resonance Sensing System

This project is an integrated system combining **physical simulation, experimental data acquisition, deep learning model training, and intelligent interactive applications** to study and utilize the dynamic and electromagnetic resonance properties of metamaterials. The system establishes a closed-loop workflow from theory to application by constructing physical models, acquiring real electromagnetic signals, training neural networks, and ultimately enabling intelligent control based on gesture recognition.

The project consists of five core Python scripts, each corresponding to a different module within the system. These modules work together to achieve real-time perception and intelligent response to metamaterial states.

---

## 1. `ElectromagneticResonanceSimulation.py`: Electromagnetic Resonance Signal Simulation

### Function Overview
This script **simulates electromagnetic resonance signals of metamaterials under different states**. By establishing a differential equation model of an LC resonant circuit, it simulates the dynamic response of a metamaterial system composed of five switchable inductor units and capacitors under various switching combinations. Its primary purposes include:
- Generating **synthetic datasets** for training neural networks.
- Validating the correctness of the physical model.
- Studying the impact of different parameters (e.g., capacitance, inductance variation coefficient) on the resonance spectrum.

### Implementation
1. **Physical Model**:
   - Based on Kirchhoff's laws and Ohm's law, a system of coupled differential equations is established, incorporating 5 inductors (L), 5 capacitors (C), and 5 resistors (R).
   - The `scipy.integrate.odeint` function is used to solve the differential equations, yielding time-series voltage and current data.

2. **State Encoding**:
   - Supports 32 different switching states (`i_type` from 0 to 31), each represented by a 5-bit binary number.
   - A binary string is generated using `bin(i_type + 0xf000)[-5:]` to control each inductor unit's state as "ON," "OFF," or "near ON."

3. **Parameter Perturbation**:
   - Introduces randomness to simulate manufacturing tolerances in real components. Inductance values are randomly generated according to a normal distribution `N(μ, σ)`, where `σ` is related to the resonance frequency variation coefficient `sigma_f`.
   - Capacitance values can also be systematically shifted using the `delta_f` parameter.

4. **Signal Processing**:
   - Applies Fast Fourier Transform (FFT) to the time-domain signal to obtain the frequency-domain amplitude spectrum.
   - Retains only spectral data within the 10 kHz to 300 kHz range (406 points), serving as input for subsequent analysis or training.

5. **Data Generation**:
   - Iterates through all states and parameter combinations to generate a large volume of simulation data, saved in `.npy` format to provide a foundation for neural network training.

---

## 2. `CollectSaveDataFromOscilloscope.py`: Oscilloscope Data Acquisition

### Function Overview
This script **acquires electromagnetic resonance signals in real-time from a UNI-T oscilloscope** using the SCPI (Standard Commands for Programmable Instruments) protocol and saves the raw data as `.npy` files. It serves as the bridge between physical experiments and digital processing, used to obtain real-world data.

### Implementation
1. **Instrument Communication**:
   - Uses the `pyvisa` library to establish a USB connection with the oscilloscope, sending SCPI commands for configuration and data retrieval.
   - Configures trigger mode (external trigger, rising edge), coupling method (DC), timebase (100 μs/div), and channel scaling (100 mV/div).

2. **Data Acquisition**:
   - Sets the waveform source to Channel 1, read mode to RAW, with a resolution of 7K points.
   - Uses the `read_raw()` method to obtain raw byte streams, converting them into NumPy arrays via `np.frombuffer`.

3. **Channel Identification**:
   - Determines the currently active metamaterial unit (0-4) by analyzing specific delay pulses in the waveform, using the `get_channel_num` function.
   - This function judges based on preset pulse positions and thresholds, ensuring data corresponds to the correct physical channel.

4. **Signal Processing**:
   - Converts raw ADC values to voltage values (based on offset and scaling parameters).
   - Applies a Hanning window to reduce spectral leakage, then performs FFT to extract the amplitude spectrum in the 10-300 kHz band.

5. **Real-time Visualization**:
   - Uses `matplotlib` to display time-domain voltage signals and frequency-domain amplitude spectra in real-time.
   - Employs multithreading to run data acquisition and visualization in parallel, ensuring system responsiveness.

---

## 3. `TrainingConvolutionalNeuralNetworks.py`: Convolutional Neural Network Training

### Function Overview
This script **trains a 1D Convolutional Neural Network (CNN)** using simulation or real measurement data generated by the first two modules. The trained network identifies the current state of the metamaterial (i.e., the 5-bit binary combination of activated units) from the electromagnetic resonance spectrum.

### Implementation
1. **Neural Network Architecture**:
   - Implements a 1D CNN with two convolutional layers (`conv1`, `conv2`) followed by a fully connected layer (`liner`).
   - Uses ELU activation functions and batch normalization to improve training stability and performance.
   - The network outputs a 5-dimensional vector representing the probability of each unit being "ON."

2. **Dataset Class**:
   - Loads pre-saved `.npy` files containing frequency spectra and corresponding labels (binary states).
   - Splits the dataset into training and testing sets based on a random array (70% training, 30% testing).
   - Applies data normalization (scaling by 1000) to improve convergence.

3. **Training Process**:
   - Uses the Adam optimizer with a learning rate of 0.002 and Mean Squared Error (MSE) as the loss function.
   - Implements a learning rate scheduler (`StepLR`) to reduce the learning rate every 20 epochs.
   - Tracks training loss, test loss, and accuracy over epochs, saving the best model and history to disk.

4. **Model Evaluation**:
   - Evaluates the model on the test set after each epoch, calculating accuracy by comparing predicted and true states.
   - Saves the best-performing model (`best_mode_oscilloscope_c5.pth`) for use in downstream applications.

---

## 4. `MetamaterialControlCalculator.py`: Intelligent Control Application

### Function Overview
This script **demonstrates a practical application of the trained CNN** by enabling real-time control of a Windows calculator application through metamaterial status. It captures live data from the oscilloscope, uses the CNN to identify the current state, and maps the state to keyboard inputs.

### Implementation
1. **Real-time Inference**:
   - Loads the pre-trained CNN model and runs it on GPU for fast inference.
   - Continuously acquires data from the oscilloscope and feeds the frequency spectrum to the CNN.

2. **State-to-Key Mapping**:
   - Defines a 5x4 virtual keypad layout (`key_map`) where each cell corresponds to a calculator function (digits, operators, etc.).
   - The CNN output indicates which row is "active," while the oscilloscope channel identifies the column, together pinpointing a specific key.

3. **User Interface**:
   - Displays a real-time visualization of the CNN's confidence levels using `matplotlib`, with colors indicating activation probability.
   - Provides visual feedback to the user, showing which key is being selected.

4. **System Control**:
   - Uses the `pyautogui` library to simulate mouse clicks at predefined screen coordinates corresponding to calculator buttons.
   - Implements a debouncing mechanism (requiring 3 consecutive identical detections) to prevent accidental inputs.

---

## 5. `MetamaterialDynamicsSimulation.py`: Metamaterial Dynamics Simulation

### Function Overview
This script **simulates the soliton propagation dynamics within a metamaterial chain** using a mass-spring model. It incorporates the static force-displacement relationship and inter-unit coupling to study nonlinear wave behavior.

### Implementation
1. **Physical Model**:
   - Models the metamaterial as a chain of masses connected by nonlinear springs.
   - Uses interpolated force-displacement curves (`fun_fx_activate`, `fun_fx_deactivate`) derived from experimental data to define spring behavior.

2. **Differential Equations**:
   - Solves a system of second-order ODEs for the position and velocity of each mass using `scipy.integrate.odeint`.
   - Includes damping and friction forces to model energy dissipation.

3. **Visualization**:
   - Animates the soliton propagation along the chain in real-time using `matplotlib`.
   - Colors the springs based on their length, visually representing the wave's amplitude and position.

4. **Data Management**:
   - Saves simulation results to `.npy` files for later analysis and visualization.
   - Loads pre-computed results if available, avoiding redundant computation.

---

## System Integration and Workflow
The five scripts form a complete pipeline:
1. **Simulation** (`ElectromagneticResonanceSimulation.py`) generates synthetic training data.
2. **Acquisition** (`CollectSaveDataFromOscilloscope.py`) captures real-world data for validation.
3. **Training** (`TrainingConvolutionalNeuralNetworks.py`) creates a CNN model to classify metamaterial states.
4. **Application** (`MetamaterialControlCalculator.py`) uses the trained model for real-time control.
5. **Dynamics** (`MetamaterialDynamicsSimulation.py`) provides deeper physical insight into the system's behavior.

---

## Dependencies and Licenses

This project relies on the following open-source libraries. Please ensure compliance with their respective licenses when using, modifying, or distributing this code.

| Library | Purpose | License |
|--------|--------|--------|
| **NumPy** | Numerical computing, array operations, FFT | [BSD 3-Clause](https://numpy.org/license.html) |
| **SciPy** | Solving ODEs, numerical integration | [BSD 3-Clause](https://scipy.org/license.html) |
| **PyTorch** | Building and training the CNN model | [BSD 3-Clause](https://pytorch.org/docs/stable/index.html) |
| **Matplotlib** | Data visualization and plotting | [PSF (Python Software Foundation)](https://matplotlib.org/stable/users/project/license.html) |
| **PyVISA** | Communication with the oscilloscope via SCPI | [BSD 3-Clause](https://pyvisa.readthedocs.io/en/latest/overview.html#license) |
| **pyautogui** | Simulating keyboard and mouse input | [BSD 3-Clause](https://github.com/asweigart/pyautogui/blob/master/LICENSE.txt) |
| **tqdm** | Displaying progress bars during training | [MIT](https://github.com/tqdm/tqdm/blob/master/LICENSE) |

**Note**: While the core libraries used are permissively licensed (mostly BSD 3-Clause), always verify the specific version and distribution terms. The project code itself is intended for research and educational purposes. Hardware interaction (e.g., with the oscilloscope) requires appropriate drivers and setup.
