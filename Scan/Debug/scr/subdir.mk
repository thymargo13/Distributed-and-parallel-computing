################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../scr/scan.cu 

CU_DEPS += \
./scr/scan.d 

OBJS += \
./scr/scan.o 


# Each subdirectory must supply rules for building sources it contributes
scr/%.o: ../scr/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/bham/pd/packages/cuda/8.0/bin/nvcc -I/bham/pd/packages/cuda/8.0/samples/common/inc -G -g -O0 -gencode arch=compute_50,code=sm_50  -odir "scr" -M -o "$(@:%.o=%.d)" "$<"
	/bham/pd/packages/cuda/8.0/bin/nvcc -I/bham/pd/packages/cuda/8.0/samples/common/inc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


