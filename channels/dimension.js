/**
 * Dimension represents a coordinate axis for neurons. Each dimension has a unique ID
 * (allocated by the Thalamus during registration) and a name (for debugging).
 * Channels create Dimension instances with a name; the Thalamus fills in `id` when the
 * channel (or spec) is registered.
 */
export class Dimension {

	constructor(name, id = null) {
		this.id = id;       // null until Thalamus allocates one (or DB passes one in on restore)
		this.name = name;   // for debugging and name↔id lookups
	}
}
