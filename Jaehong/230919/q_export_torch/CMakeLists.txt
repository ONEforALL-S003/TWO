if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

set(TFLITE2CIRCLE_PATH $<TARGET_FILE:tflite2circle>)

nncc_find_resource(PyTorchExamples)
set(PYTORCHEXAMPLES_DIR "${PyTorchExamples_DIR}")

add_test(NAME q_implant_export_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/temp.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${NNCC_OVERLAY_DIR}/venv_2_12_1"
          "${TFLITE2CIRCLE_PATH}"
          "${PYTORCHEXAMPLES_DIR}"
          "test"
)
