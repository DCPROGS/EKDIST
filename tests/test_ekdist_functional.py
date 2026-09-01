from ekdist import ekrecord

class TestFunctional:
    def setup_method(self):
        # Initialise a new empty record
        self.rec = ekrecord.SingleChannelRecord(verbose=True)
        
        # Load SCN file: can be single file or a list
        infile = ["./tests/AChsim.scn"]
        self.rec.load_SCN_file(infile)
        # Impose non zero resolution
        self.rec.tres = 30e-6
        
    def test_record_initiated(self):
        assert self.rec
        
    def test_intervals_loaded(self):
        assert len(self.rec.itint) == self.rec.header['nint']
        
    def test_intervals_resolved(self):
        assert len(self.rec.itint)-1 > len(self.rec.rtint)

    def test_periods_set(self):
        assert len(self.rec.rtint) > len(self.rec.periods.intervals)
        
# Intervals can be: 
# (1) loaded from a file;
# (2) simulated;
# (3) loaded as a list;
# (4) etc?


