def make_cell(val):

    nothing = object()
    var = val

    def cell(val=nothing):
        global var
        if val is not nothing:
            var = val
        return var
    return cell
