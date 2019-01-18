__all__ = ['ImageProcessorBase']

from pybalu.utils import create_process_base_class

ImageProcessorBase = create_process_base_class("ImageProcessor", "processing_func", "image processing")