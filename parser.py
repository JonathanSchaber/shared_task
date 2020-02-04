"""
p = NoahParser(path_in, path_out)
p.load_file()
p.copy_to_main_file()


"""


class Parser:
    
    def __init__(self, path_in, path_out):
        self.path_in = path_in
        self.path_out = path_out

    def load_file(self):
        """Open file without reading for memory restriction reasons."""
        raise NotImplementedError

    def copy_to_main_file(self):
        """Copy the loaded file to the main file."""
        raise NotImplementedError


class NoahParser(Parser): 
    pass

class ChatmaniaParser(Parser):
    pass

class SwissCrawlParser(Parser):
    pass


p = Parser('', '')
p.load_file()
