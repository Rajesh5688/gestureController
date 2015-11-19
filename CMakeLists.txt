cmake_minimum_required (VERSION 2.6)
project (GestureRecognitionApp)
#SET( CMAKE_CXX_COMPILER "C:/MinGW/bin/g++" )
find_package(OpenCV 2.4.2 REQUIRED)

add_executable(GestureRecognitionApp 
SendKeys1.h
thesis_chamfer_training_new_updated.h
thesis_chamfer_shape_updated.cpp
SendKeys1.cpp
)
target_link_libraries(GestureRecognitionApp ${OpenCV_LIBS})
