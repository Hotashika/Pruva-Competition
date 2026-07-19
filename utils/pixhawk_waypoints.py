"""Convert a downloaded MAVLink mission to a QGroundControl WPL file."""


def mission_item_to_qgc_row(message):
    message_type = message.get_type()
    x = float(message.x)
    y = float(message.y)
    if message_type == "MISSION_ITEM_INT":
        x /= 1e7
        y /= 1e7

    values = (
        int(message.seq),
        int(message.current),
        int(message.frame),
        int(message.command),
        float(message.param1),
        float(message.param2),
        float(message.param3),
        float(message.param4),
        x,
        y,
        float(message.z),
        int(message.autocontinue),
    )
    return "\t".join(str(value) for value in values)


def mission_items_to_qgc(items):
    ordered = sorted(items, key=lambda item: int(item.seq))
    rows = ["QGC WPL 110"]
    rows.extend(mission_item_to_qgc_row(item) for item in ordered)
    return "\n".join(rows) + "\n"
