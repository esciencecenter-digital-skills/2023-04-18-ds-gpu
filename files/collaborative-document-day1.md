![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 1

2023-04-18-ds-gpu

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------


## 👮Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## 🖥 Workshop websites

* [Workshop](https://esciencecenter-digital-skills.github.io/2023-04-18-ds-gpu/)
* [Course Notes](https://carpentries-incubator.github.io/lesson-gpu-programming/)
* [Google Colab](https://colab.research.google.com/)


## 👩‍🏫👩‍💻🎓 Instructors

Alessio Sclocco, Hanno Spreeuw

## 🧑‍🙋 Helpers

Giulia Crocioni, Laura Ootes

## 🗓️ Agenda
| Time | Topic |
|--:|:---|
| 09:30 | Welcome and icebreaker |
| 09:45 | Introduction |
| 10:00 | Convolve an image with a kernel on a GPU using CuPy |
| 10:30 | Running CPU/GPU agnostic code using CuPy |
| 10:45 | Coffee break |
| 11:00 | Image processing example with CuPy |
| 12:00 | Lunch break |
| 13:00 | Image processing example with CuPy |
| 14:00 | Coffee break |
| 14:15 | Run your Python code on a GPU using Numba |
| 15:00 | Coffee break |
| 15:15 | Run your Python code on a GPU using Numba |
| 16:15 | Wrap-up |
| 16:30 | End of the day |


## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building, 
  be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .

## 🔧 Exercises
### Exercise 1
Try to convolve the NumPy array deltas with the NumPy array gauss directly on the GPU, without using CuPy arrays. If this works, it should save us the time and effort of transferring deltas and gauss to the GPU.

### Solution 1
```python=
convolve2d_gpu(deltas, gauss)
```
It is unfortunately not possible to access NumPy arrays directly from the GPU because they exist in the Random Access Memory (RAM) of the host and not in GPU memory.

### Exercise 2
Compute again the speedup achieved using the GPU, but try to take also into account the time spent transferring the data to the GPU and back.

Hint: to copy a CuPy array back to the host (CPU), use the cp.asnumpy() function.

### Solution 2
```python=
def convolve_on_GPU_including_data_transfers(image_CPU, gauss_CPU):
    image_GPU = cp.asarray(image_CPU)
    gauss_GPU = cp.asarray(gauss_CPU)
    convolved_image_on_GPU = convolve2d_gpu(image_GPU, gauss_GPU)
    image_back_to_CPU = cp.asnumpy(convolved_image_on_GPU)
    return image_back_to_CPU

convolved_and_transferred = benchmark(convolve_on_GPU_including_data_transfers, (deltas, gauss), n_repeat=10)

fastest_convolved_and_transferred = np.amin(convolved_and_transferred.gpu_times)

print(f"True GPU processing time = {fastest_convolved_and_transferred:.3e}")

print(f"The speedup factor = {2.880/fastest_convolved_and_transferred:.3e}")
```

### Exercise 3:
Combine the first two steps of image processing for astronomy, i.e. determining background characteristics e.g. through $\kappa$, $\sigma$ clipping and segmentation into a single function, that works for both CPU and GPU. 

Next, write a function for connected component labelling and source measurements on the GPU and calculate the overall speedup factor for the combined four steps of image processing in astronomy on the GPU relative to the CPU. Finally, verify your output by comparing with the previous output, using the CPU.

### Solution 3:
```python=
def first_two_steps_for_both_CPU_and_GPU(data):
    data_flat = data.ravel()
    data_clipped = kappa_sigma_clipper(data_flat)
    stddev_ = np.std(data_clipped)
    threshold = 5 * stddev_
    segmented_image = np.where(data > threshold, 1,  0)
    return segmented_image

def ccl_and_source_measurements_on_CPU(data_CPU, segmented_image_CPU):
    labelled_image_CPU = np.empty(data_CPU.shape)
    number_of_sources_in_image = label_cpu(segmented_image_CPU, 
                                       output= labelled_image_CPU)
    all_positions = com_cpu(data_CPU, labelled_image_CPU, 
                            np.arange(1, number_of_sources_in_image+1))
    all_fluxes = sl_cpu(data_CPU, labelled_image_CPU, 
                            np.arange(1, number_of_sources_in_image+1))
    return np.array(all_positions), np.array(all_fluxes)

CPU_output = ccl_and_source_measurements_on_CPU(data, \
                 first_two_steps_for_both_CPU_and_GPU(data))

timing_complete_processing_CPU =  \
    benchmark(ccl_and_source_measurements_on_CPU, (data, \
       first_two_steps_for_both_CPU_and_GPU(data)), \
       n_repeat=10)

fastest_complete_processing_CPU = \
    np.amin(timing_complete_processing_CPU.cpu_times)

print(f"The four steps of image processing for astronomy take \
{1000 * fastest_complete_processing_CPU:.3e} ms\n on our CPU.")

from cupyx.scipy.ndimage import label as label_gpu
from cupyx.scipy.ndimage import center_of_mass as com_gpu
from cupyx.scipy.ndimage import sum_labels as sl_gpu

def ccl_and_source_measurements_on_GPU(data_GPU, segmented_image_GPU):
    labelled_image_GPU = cp.empty(data_GPU.shape)
    number_of_sources_in_image = label_gpu(segmented_image_GPU, 
                                           output= labelled_image_GPU)
    all_positions = com_gpu(data_GPU, labelled_image_GPU, 
                            cp.arange(1, number_of_sources_in_image+1))
    all_fluxes = sl_gpu(data_GPU, labelled_image_GPU, 
                            cp.arange(1, number_of_sources_in_image+1))
    # This seems redundant, but we want to return ndarrays (Numpy)
    # and what we have are lists. These first have to be converted to
    # Cupy arrays before they can be converted to Numpy arrays.
    return cp.asnumpy(cp.asarray(all_positions)), \
           cp.asnumpy(cp.asarray(all_fluxes))

GPU_output = ccl_and_source_measurements_on_GPU(cp.asarray(data), \
                 first_two_steps_for_both_CPU_and_GPU(cp.asarray(data)))

timing_complete_processing_GPU =  \
    benchmark(ccl_and_source_measurements_on_GPU, (cp.asarray(data), \
       first_two_steps_for_both_CPU_and_GPU(cp.asarray(data))), \
       n_repeat=10)

fastest_complete_processing_GPU = \
    np.amin(timing_complete_processing_GPU.gpu_times)

print(f"The four steps of image processing for astronomy take \
{1000 * fastest_complete_processing_GPU:.3e} ms\n on our GPU.")

overall_speedup_factor = fastest_complete_processing_CPU/ \
                         fastest_complete_processing_GPU
print(f"This means that the overall speedup factor GPU vs CPU equals: \
{overall_speedup_factor:.3e}\n")

all_positions_agree = np.allclose(CPU_output[0], GPU_output[0])
print(f"The CPU and GPU positions agree: {all_positions_agree}\n")

all_fluxes_agree = np.allclose(CPU_output[1], GPU_output[1])
print(f"The CPU and GPU fluxes agree: {all_positions_agree}\n")
```

### Exercise 4
Write a new function ```find_all_primes_cpu_and_gpu``` that uses ```check_prime_gpu_kernel``` instead of the inner ```loop of find_all_primes_cpu```. How long does this new function take to find all primes up to 10000?

### Solution 4
```python=
def find_all_primes_cpu_and_cpu(upper):
    all_primes = []
    result = np.zeros((1))
    for number in range(upper):
        is_this_a_prime[1,1](number, result)
        if result[0]:
            all_primes.append(number)
    return all_primes

find_all_primes_cpu_and_cpu(100)
```

```python=
%timeit -n 10 -r 1 find_all_primes_cpu_and_gpu(10_000)
```


## 🧠 Collaborative Notes

Today we are going to use a [Xeon Gold 5118 CPU](https://ark.intel.com/content/www/us/en/ark/products/120473/intel-xeon-gold-5118-processor-16-5m-cache-2-30-ghz.html) and a [Nvidia RTX Titan](https://www.techpowerup.com/gpu-specs/titan-rtx.c3311) GPU on the JupyterHub nodes.

![The CPU and GPU are separate devices.](https://carpentries-incubator.github.io/lesson-gpu-programming/fig/CPU_and_GPU_separated.png)

```python=
import numpy as np
import cupy as cp
```
Convolution:

$f(t) = \sum_{n=-\infty}^{n=\infty} f(n)  g(t-n)$ 

```python=
deltas = np.zeros((2048, 2048))
deltas[8::16, 8::16] = 1
```

```python=
import pylab as pyl
# Necessary command to render a matplotlib image in a Jupyter notebook.
%matplotlib inline

# Display the image
pyl.imshow(deltas[0:32, 0:32])
pyl.show()
```

```python=
x, y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
```

```python=
gauss = np.exp(-(x**2 + y**2))
pyl.imshow(gauss)
pyl.colorbar()
```

```python=
from scipy.signal import convolve2d as convolve2d_cpu
convolved_image_on_CPU = convolve2d_cpu(deltas, gauss)
pyl.imshow(convolved_image_on_CPU[0:64, 0:64])
```

```python=
%timeit -n 1 -r 1 convolve2d_cpu(deltas, gauss)
```

```python=
deltas_gpu = cp.asarray(deltas)
gauss_gpu = cp.asarray(gauss)
```

```python=
from cupyx.scipy.signal import convolve2d as convolve2d_gpu
convolved_image_on_GPU = convolve2d_gpu(deltas_gpu, gauss_gpu)
```

```python=
%timeit -n 1 -r 1 convolve2d_gpu(deltas_gpu, gauss_gpu)
```

```python=
from cupyx.profiler import benchmark

execution_gpu = benchmark(convolve2d_gpu, (deltas_gpu, gauss_gpu), n_repeat=10)

gpu_fastest_time = np.amin(execution_gpu.gpu_times)

print(f"Fastest GPU convolution =  {gpu_fastest_time:.3e}" )
```

```python=
speedup_factor = 2.880/gpu_fastest_time

print(f"The speedup_factor = {speedup_factor:3f}")
```

```python=
np.convolve(deltas_gpu.ravel(), gauss_gpu.ravel())
```

```python=
%timeit -n 10 -r 7 np.convolve(deltas.ravel(), gauss.ravel())
```

```python=
%timeit -n 10 -r 7 np.convolve(deltas_gpu.ravel(), gauss_gpu.ravel())
```

### A real-world example: image processing for radio astronomy
Four main steps in image processing for radio astronomy
1. Determine background characteristics
$\kappa$, $\sigma$ clipping
2. Threshold -> segmentation
3. Connected component labelling
4. Source measurement

```python=
# import image
import os
from astropy.io import fits

teacher_dir = os.getenv('TEACHER_DIR')
fullpath = os.path.join(teacher_dir, 'JHL_data', 
                        'GMRT_image_of_Galactic_Center.fits')
```

```python=
with fits.open(fullpath) as hdul:
    data = hdul[0].data.byteswap().newbyteorder()
```

```python=
# Inspect image
from matplotlib.colors import LogNorm

maxim = data.max()

fig = pyl.figure(figsize = (50, 12.5))
ax = fig.add_subplot(1,1,1)
im_plot = ax.imshow(np.fliplr(data),
                    cmap=pyl.cm.gray_r, 
                    norm=LogNorm(vmin = maxim/10, vmax=maxim/100))

pyl.colorbar(im_plot, ax=ax)
```


#### Determine the background characteristics of the image

```python=
mean_ = data.mean()
max_ = np.amax(data)
stddev_ = np.std(data)
median_ = np.median(data)

print(f"mean = {mean_:.3e}, median = {median_:.3e}, \
std = {stddev_:.3e}, maximum = {max_:.3e}")
```

```python=
data_flat = data.ravel()

def kappa_sigma_clipper(data_flat):
    while True:
        med = np.median(data_flat)
        std = np.std(data_flat)
        clipped_lower = data_flat.compress(data_flat> med-3*std)
        clipped_both = clipped_lower.compress(clipped_lower<med+3*std)
        if len(clipped_both) == len(data_flat):
            break
        data_flat = clipped_both
    
    return data_flat

clipped_data = kappa_sigma_clipper(data_flat)

timing_ks_clipping_cpu = %timeit -o kappa_sigma_clipper(data_flat)
fastest_ks_clipping_cpu = timing_ks_clipping_cpu.best

print(f"Fastest CPU ks clipping time = \
       {1000 * fastest_ks_clipping_cpu:.3e} ms.")
```

```python=
clipped_mean = clipped_data.mean()
clipped_median = np.median(clipped_data)
clipped_stddev_ = clipped_data.std()
clipped_max = clipped_data.max()
print(f"mean of clipped = {clipped_mean:.3e}, median of clipped = \
{clipped_median:.3e} \n standard deviation of clipped = \
{clipped_stddev_:.3e}, maximum of clipped = {clipped_max:.3e}")
```

```python=
threshold = 5 * clipped_stddev_
segmented_image = np.where(data > threshold, 1, 0)

timing_segmentation_CPU = %timeit -o np.where(data>threshold, 1, 0)
fastest_segmentation_CPU = timing_segmentation_CPU.best
print(f"Fastest CPU segmnetation time = {1000 * fastest_segmentation_CPU} ms." )
```

#### Segment the image
```python=
from scipy.ndimage import label as label_cpu
labelled_image = np.empty(data.shape)
number_of_sources_in_image = label_cpu(segmented_image, output=labelled_image)

print(f"The number of sources in this image = {number_of_sources_in_image}")
```

#### Label the semented data
```python=
from scipy.ndimage import center_of_mass as com_cpu
from scipy.ndimage import sum_labels as sl_cpu

all_positions = com_cpu(data, labelled_image, np.arange(1, number_of_sources_in_image))
all_fluxes = sl_cpu(data, labelled_image, np.arange(1, number_of_sources_in_image))

print(f"These are the ten most luminous sources in my image: {np.sort(all_fluxes)[-10:]}")
```

#### Timing
```python=
timing_CCL_CPU = %timeit -o label_cpu(segmented_image, output=labelled_image)

fastest_CCL_CPU = timing_CCL_CPU.best

print(f"Fastest CPU CCL time = {1000 * fastest_CCL_CPU:.3e} ms.")
```

```python=
%%timeit -o
all_positions = com_cpu(data, labelled_image, range(1, number_of_sources_in_image + 1))

all_fluxes = sl_cpu(data, labelled_image, range(1, number_of_sources_in_image + 1))
#Question: This shouldn't be +1 right? Since 0 is the label of empty sky
```

```python=
timing_source_measurements_CPU = _
fastest_source_measurement_CPU = timing_source_measurements_CPU.best

print(f"Fastest CPU set of source measurements = {1000*fastest_source_measurement_CPU:.3e}")
```

### Numba
```python=
def find_all_primes(upper):
    all_primes = []
    for number in range(upper):
        prime = True
        for divisor in range(2, number//2 + 1):
            if number % divisor == 0:
                prime = False
                break
        if prime:
            all_primes.append(number)
    return all_primes

find_all_primes(100)
```

```python=
timeit -n 10 -r 7 find_all_primes(10_000) 
```

```python=
import numba as nb
jit_compiled_fap = nb.njit(find_all_primes)

%timeit -n 10 -r 7 jit_compiled_fap(10_000)
```

```python=
from numba import cuda

@cuda.jit
def is_this_a_prime(number, result):
    result[0] = number
    for divisor in range(2, number//2 + 1):
        if number % divisor == 0:
            result[0] = 0
        break
```

```python=
import numpy as np

result = np.zeros((1))
is_this_a_prime[1,1](11, result)

print(result[0])
```


```python=
import math
from numba import vectorize

@vectorize(["float64(float64, float64, float64)"], target = "cpu")
def discriminant(a, b, c):
    return math.sqrt(b**2 - 4 * a * c)

size = 10**8
a = np.random.rand(size)
b = 10 + np.random.rand(size)
c = np.random.rand(size)

%timeit -n 1 -r 1 discriminant(a, b, c)
```

```python=
@vectorize(["int32(int32)"], target="cuda")
def is_this_a_prime(number):

    for divisor in range(2, number//2 + 1):
        if number % divisor == 0:
            return 0
    return number
```

```python=
check_this_range = np.arange(10_000, dtype = np.int32)
%timeit -n 1 -r 1 is_this_a_prime(check_this_range)
```

```python=
check_this_range = np.arange(10_000_000, dtype = np.int32)
%timeit -n 1 -r 1 is_this_a_prime(check_this_range)
```

## :question: Questions

## 📚 Resources

* [HackMD Markdown Guide](https://www.markdownguide.org/tools/hackmd/)
* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [Numba User Guide](https://numba.pydata.org/numba-doc/latest/user/index.html)
