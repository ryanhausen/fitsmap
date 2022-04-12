'use strict';

function buildCustomSearch(catalogPaths, zoom){

    function searchByFnames(text, callbackResponse) {

        const searchCandidates = catalogPaths.map(c => c + text + ".cbor");
        console.log(searchCandidates);
        // console.log("candidates", searchCandidates);
        Promise.all(
            searchCandidates.map(f => {
                return fetch(f).then(response => {
                    if (response.ok){
                        //console.log(response);
                        return response.arrayBuffer();
                    } else {
                        //console.log(response);
                        return Promise.resolve(null);
                    }
                }).then(buffer => {
                    //console.log(buffer);
                    return buffer ? cbor.decodeAll(buffer) : Promise.resolve(null)
                }).then(json => {
                    //console.log(json[0]);
                    return (json && json.length) ? json[0] : Promise.resolve(null);
                })

            })
        ).then(values => {
            //console.log("values", values);
            const searchResults = values.filter(val => val).map(val => {
                //console.log("val", val, val.v);
                const yIdx = val.v.length - 3;
                const xIdx = val.v.length - 2;
                const catIdx = val.v.length - 1;
                return {
                    title:val.id + ":" + val.v[catIdx],
                    loc:[val.v[yIdx],val.v[xIdx]]
                }
            });
            // console.log(searchResults);
            callbackResponse(searchResults);
        });

        return {	//called to stop previous requests on map move
            abort: function() {
                console.log('aborted request:'+ text);
            }
        };
    }

    function searchFound(latlng, title, map){
        map.setView(latlng, zoom);
    }

    const searchControl = new L.Control.Search({
        sourceData: searchByFnames,
        textPlaceholder: "Enter Catalog ID",
        textErr: "Catalog ID not found",
        markerLocation: true,
        hideMarkerOnCollapse: true,
        moveToLocation: searchFound
    });

    return searchControl;
}
